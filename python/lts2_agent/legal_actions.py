"""Legal-action derivation from a canonical state dict, scored as set-F1 vs real options (roadmap 3.3,
design §4.3 job 2 — "legal-action prediction for free").

In STS2 the legal action set is (almost) a pure function of the visible state: each hand card's
``canPlay`` flag + target type crossed with the live hittable enemies gives the ``PlayCard`` options;
potions expand the same way by their catalog target type; ``EndTurn`` is always available in the
player's turn; a ``Choice`` state's options are the pending offered cards. :func:`derive_option_keys`
implements exactly the game's :meth:`GameHost.ListOptions` rules over the **tokenized** fields, so the
same function later runs on decoder-predicted states (M4). This module measures it on **true** states
— the upper bound.

Option identity (the set-comparison key; option order does not matter)::

    PlayCard      -> ("PlayCard", cardId, targetCombatId | None)
    EndTurn       -> ("EndTurn",)
    UsePotion     -> ("UsePotion", potionId, targetCombatId | None)
    DiscardPotion -> ("DiscardPotion", potionId)
    SelectCards   -> ("SelectCards", tuple(sorted selected cardIds))   # () for "skip"

CLI::

    python -m lts2_agent.legal_actions --corpus data/corpus --split val [--limit N]

prints overall + per-kind + per-phase exact-set / precision / recall / F1 and the top mismatch
patterns, so failures are diagnosable. Stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from . import catalog, statefmt, tokens

Key = Tuple[Any, ...]

# Potion static metadata (usage window + target type) is NOT tokenized — a potion token carries only
# its catalog index — so derivation reads it from the raw potions dump (present as data/potions.json).
_POTION_META: Dict[str, Dict[str, Any]] = {}


def _potion_meta() -> Dict[str, Dict[str, Any]]:
    global _POTION_META
    if not _POTION_META:
        path = os.path.join(catalog.DATA_DIR, "potions.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _POTION_META = {p["id"]: p for p in json.load(f)}
    return _POTION_META


# ==================================================================================================
# Derivation — canonical state -> set of option-identity keys (mirrors GameHost.ListOptions).
# ==================================================================================================


def derive_option_keys(state: Dict[str, Any]) -> Set[Key]:
    """Derive the legal option-identity set from a canonical (or raw wire) state, by the game's rules.

    Covers the world-model's in-scope phases: **Combat** (PlayCard / UsePotion / DiscardPotion /
    EndTurn) and **Choice** (SelectCards). Out-of-scope non-combat views (Reward / Map) are waived by
    the tokenizer and derive to the empty set.
    """
    cv = statefmt.as_canonical(state)
    g = cv["global"]
    phase = tokens._enum_name(g["phase"], tokens.GAME_PHASES)
    turn = tokens._enum_name(g["turnPhase"], tokens.TURN_PHASES)
    keys: Set[Key] = set()

    # A pending choice takes precedence over everything (as in ListOptions).
    if cv["pending"] is not None:
        offered = cv["cards"]["offered"]
        mn = cv["pending"]["minSelect"]
        if mn <= 1:
            for c in offered:
                keys.add(("SelectCards", (_card_id(c["cardIndex"]),)))
            if mn == 0:
                keys.add(("SelectCards", ()))
        else:
            # Multi-select exact-minimum shortcut: the game takes the first `minSelect` in wire order,
            # which the tokenizer's sorted multiset does not preserve (see MISSING_INFO below).
            sub = tuple(sorted(_card_id(c["cardIndex"]) for c in offered[:mn]))
            keys.add(("SelectCards", sub))
        return keys

    if phase != "Combat" or turn != "Play":
        return keys  # not the player's turn to act / out-of-scope room view

    eids = [cr["combatId"] for cr in cv["creatures"]
            if tokens._enum_name(cr["kind"], tokens.CREATURE_KINDS) == "enemy" and cr["active"] == 1]

    for c in cv["cards"]["hand"]:
        if c.get("canPlay") != 1:
            continue
        cid = _card_id(c["cardIndex"])
        tt = tokens._enum_name(c["targetType"], tokens.TARGET_TYPES)
        if tt == "AnyEnemy":
            for eid in eids:
                keys.add(("PlayCard", cid, eid))
        else:
            keys.add(("PlayCard", cid, None))

    meta = _potion_meta()
    for pidx in cv["potions"]:
        if pidx == 0:  # empty belt slot
            continue
        pid = _potion_id(pidx)
        raw = meta.get(pid, {})
        usage = raw.get("usage")
        if usage in ("AnyTime", "CombatOnly"):  # Automatic / None never manually usable
            if raw.get("targetType") == "AnyEnemy":
                for eid in eids:
                    keys.add(("UsePotion", pid, eid))
            else:
                keys.add(("UsePotion", pid, None))
        keys.add(("DiscardPotion", pid))

    keys.add(("EndTurn",))
    return keys


def _card_id(idx: int) -> str:
    return tokens._CARDS.id_of(idx)


def _potion_id(idx: int) -> str:
    return tokens._POTIONS.id_of(idx)


# ==================================================================================================
# Real (recorded) options -> the same identity keys, for set comparison.
# ==================================================================================================


def option_key(option: Dict[str, Any]) -> Key:
    """The identity key of one recorded wire option (its position in the array is irrelevant)."""
    kind = option.get("kind")
    if kind == "PlayCard":
        return ("PlayCard", (option.get("card") or {}).get("cardId"), option.get("targetCombatId"))
    if kind == "EndTurn":
        return ("EndTurn",)
    if kind == "UsePotion":
        return ("UsePotion", option.get("potionId"), option.get("targetCombatId"))
    if kind == "DiscardPotion":
        return ("DiscardPotion", option.get("potionId"))
    if kind == "SelectCards":
        sel = tuple(sorted(c.get("cardId") for c in option.get("selectedCards") or []))
        return ("SelectCards", sel)
    if kind == "TakeReward":
        return ("TakeReward", (option.get("card") or {}).get("cardId"))
    # Any other out-of-scope kind (ProceedFromRewards / MoveTo / ChooseBundle / ...).
    return (kind,)


def recorded_keys(options: List[Dict[str, Any]]) -> Set[Key]:
    return {option_key(o) for o in options}


# ==================================================================================================
# Scoring.
# ==================================================================================================


class Tally:
    """Accumulates TP/FP/FN and exact-set matches, with per-kind and per-phase breakdowns."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.records = 0
        self.exact = 0
        self.per_kind: Dict[str, List[int]] = {}   # kind -> [tp, fp, fn]
        self.per_phase: Dict[str, List[int]] = {}  # phase -> [records, exact, tp, fp, fn]
        self.patterns: Dict[str, int] = {}

    def _kind(self, kind: str, slot: int) -> None:
        self.per_kind.setdefault(kind, [0, 0, 0])[slot] += 1

    def add(self, derived: Set[Key], recorded: Set[Key], phase: str) -> None:
        tp = derived & recorded
        fp = derived - recorded
        fn = recorded - derived
        self.tp += len(tp)
        self.fp += len(fp)
        self.fn += len(fn)
        self.records += 1
        pp = self.per_phase.setdefault(phase, [0, 0, 0, 0, 0])
        pp[0] += 1
        pp[2] += len(tp)
        pp[3] += len(fp)
        pp[4] += len(fn)
        if not fp and not fn:
            self.exact += 1
            pp[1] += 1
        for k in tp:
            self._kind(k[0], 0)
        for k in fp:
            self._kind(k[0], 1)
            self.patterns[_classify("extra", k, derived, recorded)] = \
                self.patterns.get(_classify("extra", k, derived, recorded), 0) + 1
        for k in fn:
            self._kind(k[0], 2)
            self.patterns[_classify("missing", k, derived, recorded)] = \
                self.patterns.get(_classify("missing", k, derived, recorded), 0) + 1


_REWARD_KINDS = ("TakeReward", "ProceedFromRewards")
_COMBAT_KINDS = ("PlayCard", "EndTurn", "UsePotion", "DiscardPotion")


def _classify(direction: str, key: Key, derived: Set[Key], recorded: Set[Key]) -> str:
    """Human-readable label for one FP/FN, detecting the two known root causes specifically."""
    kind = key[0]
    other = recorded if direction == "extra" else derived
    reward_screen = any(k[0] in _REWARD_KINDS for k in recorded)
    if kind in ("PlayCard", "UsePotion") and len(key) == 3:
        # Same card/potion present with a different target on the other side => target expansion diff.
        same_id_diff_target = any(o[0] == kind and o[1] == key[1] and o[2] != key[2] for o in other)
        if same_id_diff_target:
            return f"{kind} target-expansion mismatch ({direction})"
    # Post-combat reward screen: the wire still says phase=Combat but the real options are rewards,
    # and PendingRewards is not exposed in tokens — so derivation emits combat options instead.
    if kind in _REWARD_KINDS and direction == "missing":
        return "post-combat reward screen (wire phase=Combat, rewards not tokenized)"
    if reward_screen and direction == "extra" and kind in _COMBAT_KINDS:
        return "post-combat reward screen (wire phase=Combat, rewards not tokenized)"
    if kind == "SelectCards":
        return "SelectCards multi-select subset (offered-order lost for minSelect>1)"
    return f"{direction} {kind}"


def score_corpus(root: str, split: Optional[str], limit: Optional[int]) -> Tally:
    from . import corpus
    tally = Tally()
    n = 0
    for rec in corpus.iter_records(root, split):
        st = rec.get("state") or {}
        cv = statefmt.as_canonical(st)
        phase = tokens._enum_name(cv["global"]["phase"], tokens.GAME_PHASES)
        derived = derive_option_keys(cv)
        recorded = recorded_keys(rec.get("options") or [])
        tally.add(derived, recorded, phase)
        n += 1
        if limit and n >= limit:
            break
    return tally


# ==================================================================================================
# Reporting.
# ==================================================================================================


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def format_report(tally: Tally) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"LEGAL-ACTION DERIVATION vs recorded options   ({tokens.tokenizer_signature()})")
    lines.append("=" * 78)
    prec, rec, f1 = _prf(tally.tp, tally.fp, tally.fn)
    exact_rate = tally.exact / tally.records if tally.records else 0.0
    lines.append(f"records: {tally.records}   exact-set match: {tally.exact}/{tally.records} "
                 f"= {exact_rate:.4f}")
    lines.append(f"overall  precision {prec:.5f}  recall {rec:.5f}  F1 {f1:.5f}   "
                 f"(TP {tally.tp}  FP {tally.fp}  FN {tally.fn})")
    lines.append("")
    lines.append("Per option kind:")
    lines.append(f"  {'kind':16s} {'TP':>8s} {'FP':>6s} {'FN':>6s}   {'prec':>7s} {'rec':>7s} {'F1':>7s}")
    for kind in sorted(tally.per_kind):
        tp, fp, fn = tally.per_kind[kind]
        p, r, f = _prf(tp, fp, fn)
        lines.append(f"  {kind:16s} {tp:8d} {fp:6d} {fn:6d}   {p:7.4f} {r:7.4f} {f:7.4f}")
    lines.append("")
    lines.append("Per phase:")
    lines.append(f"  {'phase':10s} {'records':>8s} {'exact':>8s}   {'prec':>7s} {'rec':>7s} {'F1':>7s}")
    for phase in sorted(tally.per_phase):
        n, ex, tp, fp, fn = tally.per_phase[phase]
        p, r, f = _prf(tp, fp, fn)
        lines.append(f"  {phase:10s} {n:8d} {ex/n if n else 0:8.4f}   {p:7.4f} {r:7.4f} {f:7.4f}")
    lines.append("")
    lines.append("Top mismatch patterns:")
    if not tally.patterns:
        lines.append("  (none — exact on every record)")
    for pat, cnt in sorted(tally.patterns.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"  {cnt:7d}  {pat}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Legal-action derivation set-F1 vs recorded options.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    tally = score_corpus(args.corpus, args.split, args.limit)
    print(format_report(tally))
    return 0


if __name__ == "__main__":
    sys.exit(main())
