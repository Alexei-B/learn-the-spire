"""Closed, well-defined combat scenarios for inspecting the learned policy's decisions.

Unlike the random-fight eval, each scenario here fixes the character, the exact deck (so the opening
hand is known), and the encounter — reproducible situations where the "right" play is obvious, so we
can see *what the model does* and *what it sees* (the per-option features) when it errs.

Run::

    python -m lts2_agent.closed_eval --ckpt checkpoints/scenario
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from .env import Lts2Env, default_host_command
from .policies import jax_policy

_COMBAT_KINDS = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}

# Each: a fixed character + deck (opening hand) + encounter, with the intuitively-correct play.
SCENARIOS: list[dict[str, Any]] = [
    {
        # Dominance argument (no circumstance needed): a useful Bodyguard strictly beats a wasted Defend.
        "name": "necro-bodyguard-synergy",
        "character": "NECROBINDER",
        "cards": ["DEFEND_NECROBINDER", "DEFEND_NECROBINDER", "BODYGUARD", "UNLEASH", "STRIKE_NECROBINDER"],
        "encounter": "CultistsNormal",
        "note": "Bodyguard summons Osty: buffs Unleash's damage and blocks next turn. Play Bodyguard "
                "FIRST, then Unleash/Strike. Playing Unleash before Bodyguard, or a wasted Defend, is worse.",
    },
    {
        # Free lethal (enemy HP fixed): cultist 1 is at 4 HP, so a 6-damage Strike kills it. Removing an
        # enemy is unambiguously right regardless of anything else this turn.
        "name": "ironclad-take-lethal",
        "character": "IRONCLAD",
        "cards": ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
        "encounter": "CultistsNormal",
        "enemy_hp": [4, 999],
        "note": "Cultist 1 is at 4 HP; any attack (Strike=6) kills it. Attack cultist 1 — a free kill "
                "removes its future turns. Blocking or hitting the 999-HP cultist is strictly worse.",
    },
    {
        # AoE value: Sow hits every enemy, so its real damage is 8×#enemies and it kills the weak slime.
        # A single-target Strike/Unleash is much worse. Tests the multi-target representation.
        "name": "necro-sow-aoe",
        "character": "NECROBINDER",
        "cards": ["SOW", "STRIKE_NECROBINDER", "STRIKE_NECROBINDER", "UNLEASH", "DEFEND_NECROBINDER"],
        "encounter": "SlimesNormal",
        "enemy_hp": [3, 26, 30],
        "note": "Sow is AoE (8 to EVERY enemy): kills the 3-HP slime AND chunks the rest (~24+ total). "
                "A single Strike (6) or Unleash (7) is far worse. Sow is the clear best play.",
    },
    {
        # Focus-fire the killable enemy (enemy HP fixed): one is at 5 HP, the other 40. Kill the weak one.
        "name": "ironclad-focus-weakest",
        "character": "IRONCLAD",
        "cards": ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
        "encounter": "CultistsNormal",
        "enemy_hp": [5, 40],
        "note": "Cultist 1 (5 HP) dies to one Strike; cultist 2 (40 HP) does not. Focus the 5-HP enemy to "
                "remove it this turn rather than splitting damage into the 40-HP one.",
    },
]


def _opt_desc(o: dict[str, Any]) -> str:
    if o.get("kind") != "PlayCard":
        return o.get("kind", "?")
    c = o.get("card") or {}
    tgt = o.get("targetCombatId")
    return (f"{c.get('cardId')}{('→' + str(tgt)) if tgt is not None else '':<6} "
            f"[dmg={c.get('damage')} blk={c.get('block')} smn={c.get('summon')} "
            f"cost={c.get('energyCost')} {c.get('type')}]")


def _print_state(obs: dict[str, Any]) -> None:
    st = obs["state"]
    pcs = (st.get("players") or [{}])[0].get("combatState") or {}
    hand = [c.get("cardId") for c in pcs.get("hand") or []]
    print(f"  hand: {hand}", file=sys.stderr)
    print(f"  energy: {pcs.get('energy')}/{pcs.get('maxEnergy')}  "
          f"osty: {pcs.get('osty')}", file=sys.stderr)
    for e in (st.get("combat") or {}).get("enemies") or []:
        intents = [(i.get("type"), i.get("damage"), i.get("hits")) for i in e.get("intents") or []]
        print(f"  enemy {e.get('monsterId')} hp={e.get('currentHp')} intents={intents}", file=sys.stderr)


def _ranked(policy, state, options):
    scores = {i: s for i, s in ((e["index"], e["score"]) if isinstance(e, dict) else e
                                for e in policy(state, options))}
    rows = [(i, scores.get(i, float("-inf"))) for i, o in enumerate(options)
            if o.get("kind") in _COMBAT_KINDS]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def inspect(env: Lts2Env, policy, scn: dict[str, Any]) -> None:
    print(f"\n=== {scn['name']} ({scn['character']} vs {scn['encounter']}) ===", file=sys.stderr)
    print(f"  note: {scn['note']}", file=sys.stderr)
    obs = env.reset_combat(seed="CLOSED", character=scn["character"], cards=scn["cards"],
                           encounter=scn["encounter"], enemy_hp=scn.get("enemy_hp"))
    _print_state(obs)

    print("\n  opening option scores (combat moves, best first):", file=sys.stderr)
    for i, s in _ranked(policy, obs["state"], obs["options"]):
        print(f"    {s:8.3f}  {_opt_desc(obs['options'][i])}", file=sys.stderr)

    print("\n  turn the model actually plays:", file=sys.stderr)
    for _ in range(12):
        if obs["done"] or not obs["options"]:
            break
        if obs["state"].get("phase") != "Combat":
            break
        ranked = _ranked(policy, obs["state"], obs["options"])
        if not ranked:
            break
        idx = ranked[0][0]
        print(f"    -> {_opt_desc(obs['options'][idx])}", file=sys.stderr)
        if obs["options"][idx].get("kind") == "EndTurn":
            break
        obs = env.step(idx)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect the learned policy on closed combat scenarios.")
    p.add_argument("--ckpt", default="checkpoints/scenario")
    p.add_argument("--only", default=None, help="run only the scenario with this name")
    args = p.parse_args(argv)

    policy = jax_policy.make_policy(args.ckpt)
    with Lts2Env(host_command=default_host_command()) as env:
        for scn in SCENARIOS:
            if args.only and scn["name"] != args.only:
                continue
            inspect(env, policy, scn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
