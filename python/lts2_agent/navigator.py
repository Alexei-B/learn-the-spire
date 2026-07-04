"""A small, deterministic scripted policy for the non-combat phases of a run.

The learned net only decides **combat** (see the plan); everything else — map routing, rewards,
events, rest, shop, treasure, card choices — is driven here so a run always advances to termination.
This is a Python port of the phase switch in ``tests/Lts2.Harness.Tests/AutoPlayer.cs`` (a proven
full-run driver), operating on the observation dict + options list and returning an **option index**.

``choose`` handles every phase (delegating combat to the reference heuristic) so it doubles as a
complete standalone driver for testing; the trainer calls :func:`noncombat_action` only, and lets the
net own combat.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

from . import features
from .policies import heuristic


def combat_action(state: dict[str, Any], options: list[dict[str, Any]]) -> int:
    """A greedy combat pick from the reference heuristic (fallback: first legal option)."""
    ranking = heuristic.policy(state, options)
    if ranking:
        return max(ranking, key=lambda pair: pair[1])[0]
    return 0


def choose(state: dict[str, Any], options: list[dict[str, Any]]) -> int:
    """Pick a legal option index for any phase (combat via the heuristic)."""
    if features.is_combat(state):
        return combat_action(state, options)
    return noncombat_action(state, options)


def noncombat_action(state: dict[str, Any], options: list[dict[str, Any]]) -> int:
    """Pick a legal option index for a non-combat phase."""
    phase = state.get("phase")
    handlers = {
        "Reward": _reward,
        "Map": _map,
        "Event": _event,
        "RestSite": _rest,
        "Shop": _shop,
        "Treasure": _treasure,
        "Choice": _choice,
        "BundleChoice": _bundle,
        "CrystalSphere": _crystal_sphere,
    }
    handler = handlers.get(phase)
    if handler is not None:
        idx = handler(state, options)
        if idx is not None:
            return idx
    # Unknown/degenerate phase: take the first legal option to keep the run moving.
    return 0


# --- Per-phase handlers (return an option index, or None to fall through) ---------------------------

def _first_kind(options: list[dict[str, Any]], kind: str) -> Optional[int]:
    for i, o in enumerate(options):
        if o.get("kind") == kind:
            return i
    return None


def _reward(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    players = state.get("players") or [{}]
    potion_slot_free = any(p is None for p in (players[0].get("potions") or []))
    # Take card/relic/gold; only take a potion when a slot is free (else it's a no-op that loops).
    for i, o in enumerate(options):
        if o.get("kind") != "TakeReward":
            continue
        if not potion_slot_free and str(o.get("description", "")).startswith("Take potion"):
            continue
        return i
    return _first_kind(options, "ProceedFromRewards")


def _event(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    return _first_kind(options, "ChooseEventOption")


def _treasure(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    return _first_kind(options, "TakeTreasureRelic")


def _bundle(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    return _first_kind(options, "ChooseBundle")


def _choice(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    # Resolve a mid-effect card choice with the first offered selection (matches AutoPlayer).
    return 0 if options else None


def _rest(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    players = state.get("players") or [{}]
    hurt = (players[0].get("currentHp") or 0) < (players[0].get("maxHp") or 0)
    heal = smith = None
    for i, o in enumerate(options):
        if o.get("restOptionId") == "HEAL":
            heal = i
        elif o.get("restOptionId") == "SMITH":
            smith = i
    if hurt and heal is not None:
        return heal
    if smith is not None:
        return smith
    return 0 if options else None


def _shop(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    buy = _first_kind(options, "BuyShopItem")
    return buy if buy is not None else _first_kind(options, "MoveTo")


def _crystal_sphere(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    # Progress the minigame by clearing a hidden cell; divinations are finite so this terminates.
    return _first_kind(options, "ClickCrystalSphereCell")


def _map(state: dict[str, Any], options: list[dict[str, Any]]) -> Optional[int]:
    moves = [(i, o) for i, o in enumerate(options) if o.get("kind") == "MoveTo"]
    if not moves:
        return None
    steered = _steer_toward(state.get("map") or {}, moves, "Boss")
    return steered if steered is not None else moves[0][0]


def _steer_toward(mp: dict[str, Any], moves: list[tuple[int, dict[str, Any]]], want: str) -> Optional[int]:
    """Among reachable moves, choose the one whose subtree reaches ``want`` in the fewest steps."""
    points = mp.get("points") or []
    if not points:
        return None
    by_coord = {(p["coord"]["col"], p["coord"]["row"]): p for p in points}

    best_idx, best_dist = None, None
    for opt_idx, move in moves:
        coord = move.get("coord")
        if not coord:
            continue
        start = (coord["col"], coord["row"])
        dist = _distance_to_type(by_coord, start, want)
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist, best_idx = dist, opt_idx
    return best_idx


def _distance_to_type(by_coord: dict[tuple[int, int], dict[str, Any]],
                     start: tuple[int, int], want: str) -> Optional[int]:
    seen = {start}
    q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    while q:
        coord, dist = q.popleft()
        point = by_coord.get(coord)
        if point is None:
            continue
        if point.get("pointType") == want:
            return dist
        for child in point.get("children") or []:
            c = (child["col"], child["row"])
            if c not in seen:
                seen.add(c)
                q.append((c, dist + 1))
    return None
