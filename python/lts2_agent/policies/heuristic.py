"""A small example policy that reads the observation and picks a combat move.

It is intentionally simple — enough to prove the round-trip and show how to read the observation
schema, not a strong player. It roughly follows the built-in C# ``RulesDecisionEngine`` ordering:
secure a lethal hit, then block when a hit is coming, then attack, then play a power/skill, else end
the turn. Outside combat it declines (returns an empty ranking) so the game's own default applies.

A "policy" here is ``policy(state, options) -> ranking`` — see :mod:`lts2_agent.decision_server`.
"""

from __future__ import annotations

from typing import Any

# Priority bands, mirroring the C# rules engine: each strictly dominates the ones below it.
LETHAL = 400.0
BLOCK = 300.0
ATTACK = 200.0
POWER = 150.0
SKILL = 100.0
END_TURN = 0.0
JUNK = -1.0


def _incoming_damage(state: dict[str, Any]) -> int:
    combat = state.get("combat")
    if not combat:
        return 0
    total = 0
    for enemy in combat.get("enemies", []):
        for intent in enemy.get("intents", []):
            dmg = intent.get("damage")
            if dmg:
                total += dmg * (intent.get("hits") or 1)
    return total


def _score_option(option: dict[str, Any], state: dict[str, Any], unblocked: int) -> float:
    kind = option.get("kind")
    if kind == "EndTurn":
        return END_TURN
    if kind != "PlayCard":
        return JUNK  # potions etc.: let the game default handle them

    card = option.get("card") or {}
    card_type = card.get("type")
    damage = card.get("damage") or 0
    block = (card.get("block") or 0) + (card.get("summon") or 0)

    # Lethal: this attack (at its chosen target) would drop the enemy.
    target_id = option.get("targetCombatId")
    if card_type == "Attack" and target_id is not None and damage > 0:
        for enemy in (state.get("combat") or {}).get("enemies", []):
            if enemy.get("combatId") == target_id:
                if damage >= (enemy.get("currentHp") or 0) + (enemy.get("block") or 0):
                    return LETHAL
                break

    # Block: worth playing only when a real hit is coming.
    if block > 0 and unblocked > 0:
        return BLOCK + min(block, 50) * 0.001

    if card_type == "Attack" and damage > 0:
        return ATTACK + min(damage, 50) * 0.001
    if card_type == "Power":
        return POWER
    if card_type == "Skill":
        return SKILL
    return JUNK


def policy(state: dict[str, Any], options: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """Rank the legal ``options`` for ``state``; empty ranking = decline (out of combat)."""
    if state.get("phase") != "Combat":
        return []

    players = state.get("players") or []
    block = players[0].get("block", 0) if players else 0
    unblocked = max(0, _incoming_damage(state) - block)

    return [(i, _score_option(opt, state, unblocked)) for i, opt in enumerate(options)]
