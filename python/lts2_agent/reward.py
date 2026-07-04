"""Reward shaping for the combat policy, computed from two consecutive agent-decision observations.

The learned policy only acts in combat, so a run is a *semi-MDP* whose decision points are the combat
steps; the reward for a decision is accumulated over the whole gap until the next decision (which may
span the enemy turn, the rest of the combat, and scripted non-combat steps). The signal is the one the
plan calls for — **HP retained + enemies killed + floor progress + win bonus** — with survival (HP)
weighted highest.

All HP/damage deltas are normalized by the player's max HP so magnitudes stay ~O(1) regardless of the
character's health pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import features


@dataclass(frozen=True)
class RewardWeights:
    hp: float = 1.0          # per fraction-of-maxHP change in player HP (dominant; survival)
    damage: float = 0.5      # per fraction-of-maxHP damage dealt to enemies
    kill: float = 0.5        # per enemy killed
    floor: float = 0.5       # per floor climbed
    win: float = 10.0        # terminal victory bonus
    death: float = 5.0       # terminal death penalty


DEFAULT_WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class ScenarioWeights:
    """Reward for an isolated combat scenario (see :func:`scenario_reward` / :func:`scenario_dense_reward`)."""
    win: float = 1.0     # terminal bonus for winning the fight
    loss: float = 1.0    # terminal penalty for losing (dying)
    hp: float = 1.0      # penalty per fraction of starting HP lost (per step in the dense reward)
    # Dense shaping (per decision). damage/kill are a denser proxy for the win and are *ramped* down over
    # training (strong early to bootstrap, weak late so the policy optimizes the true win/HP objective).
    # step_penalty is always on: it charges a small cost per decision so the policy is pushed to end fights
    # efficiently rather than stalling (terminal-only reward let fight length balloon ~7x).
    damage: float = 0.5          # per fraction-of-startHP damage dealt to enemies (ramped)
    kill: float = 0.2            # per enemy killed (ramped)
    step_penalty: float = 0.02   # per decision (always on; anti-stall)


DEFAULT_SCENARIO_WEIGHTS = ScenarioWeights()


def scenario_reward(obs: dict[str, Any], start_hp: int,
                    w: ScenarioWeights = DEFAULT_SCENARIO_WEIGHTS) -> float:
    """Terminal reward for a combat scenario: win/loss outcome minus HP lost during the fight.

    Non-terminal steps get 0 (the outcome is only known when the fight ends); GAE propagates this
    terminal reward back over the fight's decisions. ``hpLost`` from the observation already has the
    character's end-of-combat starter heal added back (e.g. Ironclad's Burning Blood +6), so it
    measures real combat damage. Returns 0 until the combat is over.
    """
    info = obs["info"]
    if not info.get("combatOver"):
        return 0.0
    won = bool(info.get("won"))
    hp_lost = info.get("hpLost") or 0
    r = (w.win if won else -w.loss)
    r -= w.hp * (hp_lost / max(1, start_hp))
    return float(r)


def scenario_dense_reward(prev: dict[str, Any], cur: dict[str, Any], start_hp: int,
                          w: ScenarioWeights = DEFAULT_SCENARIO_WEIGHTS,
                          shaping_coef: float = 1.0) -> float:
    """Per-decision dense reward for a combat scenario, computed from consecutive observations.

    Always-on core: the terminal win/loss outcome, the HP lost *this step* (dense credit assignment
    instead of one lump at the end), and a small per-decision step penalty that discourages stalling.
    Ramped shaping (scaled by ``shaping_coef`` in [0, 1]): damage dealt to enemies + kills — a denser
    proxy for winning that guides early learning and is annealed away so the final policy optimizes the
    true objective. Unlike :func:`scenario_reward` this is non-zero on *every* step.
    """
    r = 0.0
    if cur.get("done") and cur["info"].get("combatOver"):
        r += w.win if cur["info"].get("won") else -w.loss

    # Dense HP: penalize HP lost this step (only losses — the post-combat starter heal shouldn't reward).
    hp_lost_step = max(0, _player_hp(prev) - _player_hp(cur)) / max(1, start_hp)
    r -= w.hp * hp_lost_step
    r -= w.step_penalty

    if shaping_coef > 0.0:
        prev_ehp = _enemy_hp(prev) if features.is_combat(prev["state"]) else 0
        cur_ehp = _enemy_hp(cur) if features.is_combat(cur["state"]) else 0
        dmg = max(0, prev_ehp - cur_ehp) / max(1, start_hp)
        prev_n = _enemy_count(prev) if features.is_combat(prev["state"]) else 0
        cur_n = _enemy_count(cur) if features.is_combat(cur["state"]) else 0
        kills = max(0, prev_n - cur_n)
        r += shaping_coef * (w.damage * dmg + w.kill * kills)
    return float(r)


def _player_hp(obs: dict[str, Any]) -> int:
    return sum(p.get("currentHp") or 0 for p in obs["info"].get("players") or [])


def _player_max_hp(obs: dict[str, Any]) -> int:
    return max(1, sum(p.get("maxHp") or 0 for p in obs["info"].get("players") or []))


def _enemy_hp(obs: dict[str, Any]) -> int:
    return sum(e.get("currentHp") or 0 for e in features._live_enemies(obs["state"]))


def _enemy_count(obs: dict[str, Any]) -> int:
    return len(features._live_enemies(obs["state"]))


def compute(prev: dict[str, Any], cur: dict[str, Any], w: RewardWeights = DEFAULT_WEIGHTS) -> float:
    """Reward for the transition from decision ``prev`` to the next decision/terminal ``cur``."""
    max_hp = _player_max_hp(prev)

    hp_delta = (_player_hp(cur) - _player_hp(prev)) / max_hp          # <0 when damaged

    prev_in_combat = features.is_combat(prev["state"])
    cur_in_combat = features.is_combat(cur["state"])
    prev_enemy_hp = _enemy_hp(prev) if prev_in_combat else 0
    cur_enemy_hp = _enemy_hp(cur) if cur_in_combat else 0
    # Damage we dealt: drop in enemy HP; if the combat ended, the remaining HP was cleared (a kill).
    enemy_dealt = max(0, prev_enemy_hp - cur_enemy_hp) / max_hp

    prev_enemies = _enemy_count(prev) if prev_in_combat else 0
    cur_enemies = _enemy_count(cur) if cur_in_combat else 0
    kills = max(0, prev_enemies - cur_enemies)

    r = w.hp * hp_delta + w.damage * enemy_dealt + w.kill * kills

    floors = (cur["info"].get("floor", 0) - prev["info"].get("floor", 0))
    if floors > 0:
        r += w.floor * floors

    if cur.get("done"):
        r += w.win if cur["info"].get("victory") else -w.death
    return float(r)
