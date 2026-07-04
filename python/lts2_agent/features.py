"""Shared feature encoding for the learned combat policy — the train/serve parity contract.

Both the trainer (:mod:`lts2_agent.train`) and the served policy
(:mod:`lts2_agent.policies.jax_policy`) featurize observations through *this* module, so a net
trained against the environment sees byte-identical inputs when it runs behind the decision server.
Keep it NumPy-only (no JAX) so it imports everywhere, and keep every vector a **fixed length** so the
model's input dims never shift.

The encoding is deliberately combat-focused (the net only decides combat — see the plan): the state
vector summarizes the battlefield, and each option is scored from its own feature row plus a stable
hashed **card-id bucket** (an index into the model's embedding table). Card ids are strings with no
integer enum, so we hash them with a *stable* CRC (never Python's per-process ``hash``) to guarantee
the same card maps to the same bucket in training and in the TUI.
"""

from __future__ import annotations

import zlib
from typing import Any, Optional

import numpy as np

# --- Vocab / sizes ---------------------------------------------------------------------------------

# Bump when the encoding layout changes so stale checkpoints are rejected instead of silently misread.
FEATURE_VERSION = 1

CARD_VOCAB = 4096          # hashed card-id embedding table size (bucket 0 = "no card")
POWER_BUCKETS = 16         # hashed power-id buckets, per creature side
MAX_OPTIONS = 64           # padded action-set width (measured max combat option count ~21)

# Card-type one-hot order (the game's CardType enum, serialized as its name).
_CARD_TYPES = ["Attack", "Skill", "Power", "Status", "Curse", "Quest"]
# Option-kind one-hot order (only combat kinds matter; others fall into "Other").
_OPTION_KINDS = ["PlayCard", "EndTurn", "UsePotion", "DiscardPotion"]


def _bucket(text: str, n: int) -> int:
    """A stable (cross-process) hash of ``text`` into ``[0, n)`` via CRC32."""
    return zlib.crc32(text.encode("utf-8")) % n


def card_bucket(card: Optional[dict[str, Any]]) -> int:
    """The embedding index for a card: 0 for no card, else 1 + a stable hash of (id, upgraded)."""
    if not card:
        return 0
    key = card.get("cardId", "") + ("+" if card.get("upgraded") else "")
    return 1 + _bucket(key, CARD_VOCAB - 1)


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
    return np.concatenate(parts).astype(np.float32)


# --- Option encoding -------------------------------------------------------------------------------

def _weakest_id(state: dict[str, Any]) -> Optional[int]:
    live = _live_enemies(state)
    if not live:
        return None
    return min(live, key=lambda e: e.get("currentHp") or 0).get("combatId")


def _option_scalars(option: dict[str, Any], state: dict[str, Any],
                    enemies_by_id: dict[int, dict[str, Any]], weakest_id: Optional[int]) -> list[float]:
    kind = option.get("kind", "")
    kind_oh = [1.0 if kind == k else 0.0 for k in _OPTION_KINDS]

    card = option.get("card") or {}
    ctype = card.get("type")
    type_oh = [1.0 if ctype == t else 0.0 for t in _CARD_TYPES]

    damage = card.get("damage") or 0
    block = card.get("block") or 0
    summon = card.get("summon") or 0

    target_id = option.get("targetCombatId")
    target = enemies_by_id.get(target_id) if target_id is not None else None
    target_hp = (target.get("currentHp") or 0) if target else 0
    target_block = (target.get("block") or 0) if target else 0
    is_targeted = 1.0 if target_id is not None else 0.0
    lethal = 1.0 if (target is not None and damage >= target_hp + target_block and damage > 0) else 0.0
    is_weakest = 1.0 if (target_id is not None and target_id == weakest_id) else 0.0

    return [
        *kind_oh,
        *type_oh,
        (card.get("energyCost") or 0) / 6.0,
        1.0 if card.get("costsX") else 0.0,
        damage / 30.0,
        block / 30.0,
        summon / 30.0,
        (card.get("starCost") or 0) / 6.0,
        1.0 if card.get("upgraded") else 0.0,
        (card.get("replayCount") or 0) / 3.0,
        is_targeted,
        lethal,
        is_weakest,
        target_hp / 100.0,
        target_block / 30.0,
    ]


def encode_options(state: dict[str, Any], options: list[dict[str, Any]]):
    """Encode the legal options into fixed-width padded arrays.

    Returns ``(dense, card_idx, mask)``:
      * ``dense``    — ``float32[MAX_OPTIONS, OPTION_DIM]`` per-option features (padding rows are 0),
      * ``card_idx`` — ``int32[MAX_OPTIONS]`` card embedding buckets (0 for no-card / padding),
      * ``mask``     — ``bool[MAX_OPTIONS]`` True for a real, legal option.

    Options beyond ``MAX_OPTIONS`` are dropped (measured combat max ~21, so this is slack).
    """
    enemies_by_id = {e.get("combatId"): e for e in _enemies(state)}
    weakest_id = _weakest_id(state)

    dense = np.zeros((MAX_OPTIONS, OPTION_DIM), dtype=np.float32)
    card_idx = np.zeros(MAX_OPTIONS, dtype=np.int32)
    mask = np.zeros(MAX_OPTIONS, dtype=bool)

    for i, opt in enumerate(options[:MAX_OPTIONS]):
        dense[i] = np.asarray(_option_scalars(opt, state, enemies_by_id, weakest_id), dtype=np.float32)
        card_idx[i] = card_bucket(opt.get("card"))
        mask[i] = True
    return dense, card_idx, mask


# --- Derived dims (computed once from a canonical sample so they can never drift) -------------------

_SAMPLE_STATE: dict[str, Any] = {
    "phase": "Combat",
    "players": [{"currentHp": 1, "maxHp": 1, "block": 0, "combatState": {"energy": 0, "maxEnergy": 1}}],
    "combat": {"enemies": []},
}
STATE_DIM = int(encode_state(_SAMPLE_STATE).shape[0])
OPTION_DIM = len(_option_scalars({"kind": "EndTurn"}, _SAMPLE_STATE, {}, None))
