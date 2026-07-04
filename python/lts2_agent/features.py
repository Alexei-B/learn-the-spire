"""Shared feature encoding for the learned combat policy — the train/serve parity contract.

Both the trainer (:mod:`lts2_agent.train`) and the served policy
(:mod:`lts2_agent.policies.jax_policy`) featurize observations through *this* module, so a net
trained against the environment sees byte-identical inputs when it runs behind the decision server.
Keep it NumPy-only (no JAX) so it imports everywhere, and keep every vector a **fixed length** so the
model's input dims never shift.

Each card (whether in hand or as an option) is described by a shared :func:`_card_features` vector plus
a stable hashed **card-id bucket** (an index into the model's embedding table). Crucially the whole
**hand** is encoded (rich features + embeddings), not just its size — so the policy knows what it is
holding and can learn card *synergies* (play a summon/buff before its payoff, etc.) generally, rather
than scoring each option in isolation. Card ids are strings with no integer enum, so we hash them with
a *stable* CRC (never Python's per-process ``hash``) so the same card maps to the same bucket everywhere.
"""

from __future__ import annotations

import zlib
from typing import Any, Optional

import numpy as np

from . import card_catalog

# --- Vocab / sizes ---------------------------------------------------------------------------------

# Bump when the encoding layout changes so stale checkpoints are rejected instead of silently misread.
FEATURE_VERSION = 5

# Features are designed to be O(1), but a few quantities can accumulate without bound in a long fight
# (block under Barricade, scaling damage, stacked powers). An unclipped outlier feeds the net a huge
# input and its value head then emits enormous outputs (~1e6), blowing up returns and the value loss.
# Saturate every feature to this range; normal values sit well inside it, so it's a no-op except on the
# pathological tail.
FEATURE_CLIP = 30.0

# Card identity is a stable dense index over the game's card catalog (from card_catalog / the
# --dump-cards dump): no hash collisions, and it also keys the model's static tags/keywords/var-keys
# table. If the dump is absent (a fresh clone), fall back to a CRC32 hash into a fixed vocab.
_CATALOG = card_catalog.try_load()
CARD_VOCAB = _CATALOG.size if _CATALOG else 4096   # embedding table size (index 0 = "no card")
CARD_STATIC_DIM = _CATALOG.static_dim if _CATALOG else 0
CATALOG_SIGNATURE = _CATALOG.signature if _CATALOG else "hash"

POWER_BUCKETS = 16         # hashed power-id buckets, per creature side
MAX_OPTIONS = 64           # padded action-set width (measured max combat option count ~21)
MAX_HAND = 12              # padded hand width for the hand-content encoding

# Card-type one-hot order (the game's CardType enum, serialized as its name).
_CARD_TYPES = ["Attack", "Skill", "Power", "Status", "Curse", "Quest"]
# Option-kind one-hot order (only combat kinds matter; others fall into "Other").
_OPTION_KINDS = ["PlayCard", "EndTurn", "UsePotion", "DiscardPotion"]

# The model input arrays, in the order the model's __call__ consumes them.
MODEL_KEYS = ("g", "hand_dense", "hand_idx", "hand_mask", "dense", "card_idx", "mask")


def _bucket(text: str, n: int) -> int:
    """A stable (cross-process) hash of ``text`` into ``[0, n)`` via CRC32."""
    return zlib.crc32(text.encode("utf-8")) % n


def card_index(card: Optional[dict[str, Any]]) -> int:
    """The embedding index for a card: 0 for no card, else its stable catalog index (or a CRC32 hash
    when no catalog is loaded). Keyed by base card id — the ``upgraded`` flag is a separate feature."""
    if not card:
        return 0
    card_id = card.get("cardId", "")
    if _CATALOG is not None:
        return _CATALOG.index_of(card_id)
    key = card_id + ("+" if card.get("upgraded") else "")
    return 1 + _bucket(key, CARD_VOCAB - 1)


def card_static_table() -> np.ndarray:
    """The [CARD_VOCAB, CARD_STATIC_DIM] multi-hot table (tags/keywords/var-keys) the model gathers by
    card index. Empty (width 0) when no catalog is loaded."""
    if _CATALOG is not None:
        return _CATALOG.static_table
    return np.zeros((CARD_VOCAB, 0), dtype=np.float32)


# --- Small readers (null-tolerant; the wire omits null fields) --------------------------------------

def is_combat(state: dict[str, Any]) -> bool:
    return state.get("phase") == "Combat" and bool(state.get("combat"))


def _players(state: dict[str, Any]) -> list[dict[str, Any]]:
    return state.get("players") or []


def _enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    combat = state.get("combat") or {}
    return combat.get("enemies") or []


def _live_enemies(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in _enemies(state) if e.get("isHittable") and (e.get("currentHp") or 0) > 0]


def incoming_damage(state: dict[str, Any]) -> int:
    """Total telegraphed attack damage this turn (Σ intent damage × hits), before block."""
    total = 0
    for e in _enemies(state):
        for intent in e.get("intents") or []:
            dmg = intent.get("damage")
            if dmg:
                total += dmg * (intent.get("hits") or 1)
    return total


def _osty_hp(player: dict[str, Any]) -> int:
    osty = (player.get("combatState") or {}).get("osty")
    return osty.get("currentHp", 0) if osty and osty.get("isAlive") else 0


def _power_buckets(powers: list[dict[str, Any]]) -> np.ndarray:
    """Aggregate a creature's powers into fixed hashed buckets (signed amount, normalized)."""
    v = np.zeros(POWER_BUCKETS, dtype=np.float32)
    for p in powers or []:
        b = _bucket(p.get("powerId", ""), POWER_BUCKETS)
        v[b] += (p.get("amount") or 0) / 10.0
    return v


# --- Per-card features (shared by the hand and the options) ----------------------------------------

def _card_features(card: dict[str, Any]) -> list[float]:
    """Describe one card's mechanics so the policy can 'understand' it — type, cost, its current
    damage/block/summon *and how amplified they are right now* (``damage - baseDamage`` captures
    circumstance/power amplifiers like Strength scaling or Gold Axe), plus star cost, upgrade, replays,
    granted keywords, and enchant/affliction flags. The game already folds powers/targets into the
    previewed damage/block, so this reads the live effect, not just the printed number."""
    type_oh = [1.0 if card.get("type") == t else 0.0 for t in _CARD_TYPES]

    damage = card.get("damage") or 0
    base_damage = card.get("baseDamage")
    base_damage = damage if base_damage is None else base_damage
    block = card.get("block") or 0
    base_block = card.get("baseBlock")
    base_block = block if base_block is None else base_block

    return [
        *type_oh,
        (card.get("energyCost") or 0) / 6.0,
        1.0 if card.get("costsX") else 0.0,
        (card.get("starCost") or 0) / 6.0,
        1.0 if card.get("upgraded") else 0.0,
        (card.get("replayCount") or 0) / 3.0,
        damage / 30.0,
        (damage - base_damage) / 30.0,        # amplified (>0) or weakened (<0) right now
        block / 30.0,
        (block - base_block) / 30.0,
        (card.get("summon") or 0) / 30.0,
        len(card.get("addedKeywords") or []) / 5.0,
        1.0 if card.get("enchantmentId") else 0.0,
        1.0 if card.get("afflictionId") else 0.0,
    ]


# --- State encoding --------------------------------------------------------------------------------

def _state_scalars(state: dict[str, Any]) -> list[float]:
    players = _players(state)
    me = players[0] if players else {}
    cs = me.get("combatState") or {}

    max_hp = max(1, me.get("maxHp") or 1)
    cur_hp = me.get("currentHp") or 0
    block = me.get("block") or 0
    energy = cs.get("energy") or 0
    max_energy = max(1, cs.get("maxEnergy") or me.get("maxEnergy") or 1)

    live = _live_enemies(state)
    enemy_hp = sum(e.get("currentHp") or 0 for e in live)
    enemy_max = sum(e.get("maxHp") or 0 for e in live)
    min_enemy_hp = min((e.get("currentHp") or 0 for e in live), default=0)
    incoming = incoming_damage(state)
    n_attackers = sum(
        1 for e in _enemies(state)
        for i in (e.get("intents") or []) if i.get("damage")
    )
    osty = _osty_hp(me)

    return [
        1.0 if is_combat(state) else 0.0,
        cur_hp / max_hp,
        cur_hp / 100.0,
        block / 50.0,
        energy / 10.0,
        max_energy / 10.0,
        energy / max_energy,
        (cs.get("stars") or 0) / 10.0,
        (cs.get("turnNumber") or 0) / 20.0,
        len(cs.get("hand") or []) / 10.0,
        len(cs.get("drawPile") or []) / 30.0,
        len(cs.get("discardPile") or []) / 30.0,
        len(cs.get("exhaustPile") or []) / 20.0,
        len(cs.get("orbs") or []) / 6.0,
        (cs.get("orbSlots") or 0) / 6.0,
        osty / max_hp,
        len(live) / 5.0,
        (enemy_hp / enemy_max) if enemy_max else 0.0,
        min_enemy_hp / 100.0,
        incoming / 50.0,
        max(0, incoming - block - osty) / 50.0,
        n_attackers / 5.0,
        (state.get("floor") or 0) / 50.0,
        (state.get("actIndex") or 0) / 3.0,
    ]


def encode_state(state: dict[str, Any]) -> np.ndarray:
    """The fixed-length global state vector ``g`` (STATE_DIM,)."""
    me = (_players(state)[0] if _players(state) else {})
    cs = me.get("combatState") or {}
    parts = [
        np.asarray(_state_scalars(state), dtype=np.float32),
        _power_buckets(cs.get("powers") or []),
        _power_buckets([p for e in _live_enemies(state) for p in (e.get("powers") or [])]),
    ]
    g = np.concatenate(parts).astype(np.float32)
    return np.clip(g, -FEATURE_CLIP, FEATURE_CLIP)


def encode_hand(state: dict[str, Any]):
    """Encode the player's whole hand as ``(dense[MAX_HAND, CARD_FEAT_DIM], idx[MAX_HAND], mask)`` — the
    per-card features + embedding bucket for each held card. Lets the policy see combos in hand."""
    cs = (_players(state)[0] if _players(state) else {}).get("combatState") or {}
    hand = cs.get("hand") or []
    dense = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)
    idx = np.zeros(MAX_HAND, dtype=np.int32)
    mask = np.zeros(MAX_HAND, dtype=bool)
    for i, card in enumerate(hand[:MAX_HAND]):
        dense[i] = np.asarray(_card_features(card), dtype=np.float32)
        idx[i] = card_index(card)
        mask[i] = True
    np.clip(dense, -FEATURE_CLIP, FEATURE_CLIP, out=dense)
    return dense, idx, mask


# --- Option encoding -------------------------------------------------------------------------------

def _weakest_id(state: dict[str, Any]) -> Optional[int]:
    live = _live_enemies(state)
    if not live:
        return None
    return min(live, key=lambda e: e.get("currentHp") or 0).get("combatId")


def _option_scalars(option: dict[str, Any], state: dict[str, Any],
                    enemies_by_id: dict[int, dict[str, Any]], weakest_id: Optional[int],
                    live_enemies: list[dict[str, Any]]) -> list[float]:
    kind = option.get("kind", "")
    kind_oh = [1.0 if kind == k else 0.0 for k in _OPTION_KINDS]

    card = option.get("card") or {}
    damage = card.get("damage") or 0
    target_type = card.get("targetType")

    target_id = option.get("targetCombatId")
    target = enemies_by_id.get(target_id) if target_id is not None else None
    target_hp = (target.get("currentHp") or 0) if target else 0
    target_block = (target.get("block") or 0) if target else 0
    is_targeted = 1.0 if target_id is not None else 0.0
    lethal = 1.0 if (target is not None and damage >= target_hp + target_block and damage > 0) else 0.0
    is_weakest = 1.0 if (target_id is not None and target_id == weakest_id) else 0.0

    # Multi-target damage: an AllEnemies attack (e.g. Sow) hits *every* live enemy, so its real value is
    # damage × #enemies and it can kill several at once — invisible if we only look at the per-hit number
    # and single-target lethality. Capture total damage dealt and total kills across the board.
    is_aoe = 1.0 if target_type == "AllEnemies" else 0.0
    if is_aoe and damage > 0:
        num_targets = len(live_enemies)
        total_damage = damage * num_targets
        kills = sum(1 for e in live_enemies
                    if damage >= (e.get("currentHp") or 0) + (e.get("block") or 0))
    else:
        num_targets = 1 if damage > 0 else 0
        total_damage = damage
        kills = lethal  # single-target: 1 if it kills its target

    return [
        *kind_oh,
        *_card_features(card),
        is_targeted,
        lethal,
        is_weakest,
        target_hp / 100.0,
        target_block / 30.0,
        is_aoe,
        num_targets / 5.0,
        total_damage / 30.0,
        kills / 5.0,
    ]


def encode_options(state: dict[str, Any], options: list[dict[str, Any]]):
    """Encode the legal options into fixed-width padded arrays ``(dense, card_idx, mask)``. Options
    beyond ``MAX_OPTIONS`` are dropped (measured combat max ~21, so this is slack)."""
    enemies_by_id = {e.get("combatId"): e for e in _enemies(state)}
    weakest_id = _weakest_id(state)
    live = _live_enemies(state)

    dense = np.zeros((MAX_OPTIONS, OPTION_DIM), dtype=np.float32)
    card_idx = np.zeros(MAX_OPTIONS, dtype=np.int32)
    mask = np.zeros(MAX_OPTIONS, dtype=bool)

    for i, opt in enumerate(options[:MAX_OPTIONS]):
        dense[i] = np.asarray(_option_scalars(opt, state, enemies_by_id, weakest_id, live), dtype=np.float32)
        card_idx[i] = card_index(opt.get("card"))
        mask[i] = True
    np.clip(dense, -FEATURE_CLIP, FEATURE_CLIP, out=dense)
    return dense, card_idx, mask


def encode(state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """All model inputs for one observation, keyed by :data:`MODEL_KEYS` (unbatched)."""
    hand_dense, hand_idx, hand_mask = encode_hand(state)
    dense, card_idx, mask = encode_options(state, options)
    return {
        "g": encode_state(state),
        "hand_dense": hand_dense, "hand_idx": hand_idx, "hand_mask": hand_mask,
        "dense": dense, "card_idx": card_idx, "mask": mask,
    }


# --- Derived dims (computed once from a canonical sample so they can never drift) -------------------

CARD_FEAT_DIM = len(_card_features({}))

_SAMPLE_STATE: dict[str, Any] = {
    "phase": "Combat",
    "players": [{"currentHp": 1, "maxHp": 1, "block": 0, "combatState": {"energy": 0, "maxEnergy": 1}}],
    "combat": {"enemies": []},
}
STATE_DIM = int(encode_state(_SAMPLE_STATE).shape[0])
OPTION_DIM = len(_option_scalars({"kind": "EndTurn"}, _SAMPLE_STATE, {}, None, []))

# Human-readable names for each column of a card / option feature vector (same order as the builders
# above), so a decision can be inspected feature-by-feature.
CARD_FEATURE_NAMES = [
    "type_Attack", "type_Skill", "type_Power", "type_Status", "type_Curse", "type_Quest",
    "cost", "costsX", "starCost", "upgraded", "replayCount",
    "damage", "dmg_amp", "block", "blk_amp", "summon", "keywords", "enchant", "afflict",
]
OPTION_FEATURE_NAMES = [
    "kind_PlayCard", "kind_EndTurn", "kind_UsePotion", "kind_DiscardPotion",
    *CARD_FEATURE_NAMES,
    "is_targeted", "lethal", "is_weakest", "target_hp", "target_block",
    "is_aoe", "num_targets", "total_damage", "kills",
]
assert len(CARD_FEATURE_NAMES) == CARD_FEAT_DIM, (len(CARD_FEATURE_NAMES), CARD_FEAT_DIM)
assert len(OPTION_FEATURE_NAMES) == OPTION_DIM, (len(OPTION_FEATURE_NAMES), OPTION_DIM)
