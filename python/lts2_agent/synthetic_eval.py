"""Fast, harness-free eval of the policy on a *fixed captured input*.

The real fights are too noisy and slow to tell whether the model has learned a specific decision. Here
we capture one observation from the harness once (``--capture``), save it as a small JSON fixture, and
thereafter evaluate the model on that exact input with **no game simulation** — just encode → forward →
rank. That makes a targeted decision (e.g. "does it play Bodyguard before Unleash and never waste a
Defend?") a millisecond check we can watch every eval step and even early-stop on.

The fixture is the raw observation (state + legal options); we re-encode it with the current
:mod:`lts2_agent.features`, so it survives feature changes and lets us print exactly what the model is
fed (``describe_input``) to spot representation bugs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import numpy as np

from . import features, model

# The Necrobinder starter-hand decision we're chasing: [Strike, Defend, Unleash, Bodyguard] vs a
# non-attacking enemy. Correct play (3 energy): Bodyguard before Unleash, never waste a Defend.
DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "evals", "necro_bodyguard.json")


def load_fixture(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def per_card_scores(apply_fn, params, obs: dict[str, Any]) -> dict[str, float]:
    """Model score per distinct move: a card's best score over its targets, plus EndTurn."""
    feats = features.encode(obs["state"], obs["options"])
    logits, _ = model.forward1(apply_fn, params, feats)
    logits = np.asarray(logits[0])
    best: dict[str, float] = {}
    for i, o in enumerate(obs["options"][:features.MAX_OPTIONS]):
        key = (o.get("card") or {}).get("cardId") if o.get("kind") == "PlayCard" else o.get("kind")
        best[key] = max(best.get(key, -1e18), float(logits[i]))
    return best


def ranking(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def bodyguard_pass(scores: dict[str, float]) -> tuple[bool, str]:
    """PASS when the ranking implies a correct starter-hand turn: Bodyguard ranked above Unleash, and
    Defend ranked below every useful card (so the greedy 3-card turn drops Defend and never plays
    Unleash before Bodyguard). Card ids are matched loosely by name."""
    def find(sub):
        for k in scores:
            if k and sub in k.upper():
                return k
        return None
    bg, un, st, df = find("BODYGUARD"), find("UNLEASH"), find("STRIKE"), find("DEFEND")
    if not all([bg, un, st, df]):
        return False, f"missing card(s): bg={bg} un={un} st={st} df={df}"
    if scores[bg] <= scores[un]:
        return False, f"Bodyguard({scores[bg]:.3f}) not above Unleash({scores[un]:.3f})"
    if scores[df] >= min(scores[bg], scores[un], scores[st]):
        return False, f"Defend({scores[df]:.3f}) not below all of Bodyguard/Unleash/Strike"
    return True, "Bodyguard>Unleash and Defend lowest"


def describe_input(obs: dict[str, Any]) -> str:
    """A human-readable dump of the observation + the labeled features the model actually receives."""
    st, opts = obs["state"], obs["options"]
    pcs = (st.get("players") or [{}])[0].get("combatState") or {}
    lines = ["--- OBSERVATION ---",
             f"phase={st.get('phase')} energy={pcs.get('energy')}/{pcs.get('maxEnergy')} "
             f"osty={pcs.get('osty')}",
             f"hand={[c.get('cardId') for c in pcs.get('hand') or []]}"]
    for e in (st.get("combat") or {}).get("enemies") or []:
        lines.append(f"  enemy {e.get('monsterId')} hp={e.get('currentHp')}/{e.get('maxHp')} "
                     f"intents={[(i.get('type'), i.get('damage'), i.get('hits')) for i in e.get('intents') or []]}")

    dense, card_idx, mask = features.encode_options(st, opts)
    lines.append("\n--- OPTION FEATURES (only non-zero shown) ---")
    for i, o in enumerate(opts):
        label = (o.get("card") or {}).get("cardId") if o.get("kind") == "PlayCard" else o.get("kind")
        tgt = o.get("targetCombatId")
        feats = {features.OPTION_FEATURE_NAMES[j]: round(float(dense[i, j]), 3)
                 for j in range(features.OPTION_DIM) if abs(dense[i, j]) > 1e-6}
        lines.append(f"  [{i}] {label}{('→'+str(tgt)) if tgt is not None else ''} bucket={int(card_idx[i])}")
        lines.append(f"       {feats}")
    return "\n".join(lines)


# --- Capture (uses the harness once) ---------------------------------------------------------------

def capture(path: str, character: str, cards: list[str], encounter: str,
            enemy_hp: Optional[list[int]] = None, host_command: Optional[list[str]] = None) -> dict[str, Any]:
    from .env import Lts2Env, default_host_command
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with Lts2Env(host_command=host_command or default_host_command()) as env:
        obs = env.reset_combat(seed="FIXTURE", character=character, cards=cards,
                               encounter=encounter, enemy_hp=enemy_hp)
    fixture = {"state": obs["state"], "options": obs["options"]}
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1)
    return fixture


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Harness-free synthetic eval of the policy on a fixture.")
    p.add_argument("--ckpt", default="checkpoints/scenario")
    p.add_argument("--fixture", default=DEFAULT_FIXTURE)
    p.add_argument("--show", action="store_true", help="print the observation + labeled model input")
    p.add_argument("--capture", action="store_true", help="(re)capture the default Necrobinder fixture")
    args = p.parse_args(argv)

    if args.capture:
        capture(args.fixture, "NECROBINDER",
                ["STRIKE_NECROBINDER", "DEFEND_NECROBINDER", "UNLEASH", "BODYGUARD"],
                "CultistsNormal", enemy_hp=[60, 60])
        print(f"captured fixture -> {args.fixture}", file=sys.stderr)

    obs = load_fixture(args.fixture)
    if args.show:
        print(describe_input(obs), file=sys.stderr)

    import jax
    m, params, _ = model.load_checkpoint(args.ckpt)
    apply = jax.jit(m.apply)
    scores = per_card_scores(apply, params, obs)
    ok, reason = bodyguard_pass(scores)
    print("\n--- RANKING (high → low) ---", file=sys.stderr)
    for name, s in ranking(scores):
        print(f"  {s:8.3f}  {name}", file=sys.stderr)
    print(f"\nPASS={ok}  ({reason})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
