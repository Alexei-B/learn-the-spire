"""CP2 corpus report: composition + realistic-deck distribution stats over a transition corpus.

Reads a corpus root (written by :mod:`lts2_agent.collect`) and prints the artifact the product owner
reviews at CP2:

* **composition** — record/fight counts and fight win-rate broken down by split / regime / policy /
  act / room / character;
* **realistic-deck stats** — histograms of #removals and #additions, the realized pool distribution of
  added cards (own / colorless / curse / off-character, from ``cards.json`` pool metadata) against the
  configured 60/25/12/3 weights, and the top-20 most-added cards;
* **a sample of 20 realistic decks** (character + final deck ids) to eyeball for "looks like act 1";
* **a determinism note** — re-derives 3 fights' splits from their seeds to show the split is stable and
  leak-free (matches the on-disk split it was filed under).

``python -m lts2_agent.corpus_report --corpus <root>`` prints human-readable text; ``--json`` emits the
same aggregation as one machine-readable JSON object on stdout. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import corpus

# The configured realistic-addition pool weights (roadmap goal 4). Surfaced here to compare realized.
CONFIGURED_WEIGHTS = {"own": 0.60, "colorless": 0.25, "curse": 0.12, "offCharacter": 0.03}

# Card pools that are not a playable character's pool.
_NONCHAR_POOLS = {"colorless", "curse", "status", "event", "quest", "token"}


# --------------------------------------------------------------------------------------------------
# Card metadata (pool classification for added cards).
# --------------------------------------------------------------------------------------------------


def load_card_meta(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load ``cards.json`` into ``{card_id: {pool, colorless, curse, category}}`` (empty if absent)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "cards.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        cards = json.load(f)
    return {c["id"]: c for c in cards}


def classify_added_card(card_id: str, character: Optional[str],
                        card_meta: Dict[str, Dict[str, Any]]) -> str:
    """Classify one added card into a pool bucket relative to ``character``.

    Returns ``own`` / ``colorless`` / ``curse`` / ``offCharacter`` / ``ownOrOff`` (character unknown) /
    ``other`` (non-deck pool) / ``unknown`` (id absent from the catalog — null-tolerant)."""
    meta = card_meta.get(card_id)
    if not meta:
        return "unknown"
    pool = meta.get("pool")
    if meta.get("curse") or meta.get("category") == "Curse" or pool == "curse":
        return "curse"
    if meta.get("colorless") or pool == "colorless":
        return "colorless"
    if pool in _NONCHAR_POOLS:
        return "other"
    # A character pool.
    if character is None:
        return "ownOrOff"
    return "own" if str(pool).lower() == str(character).lower() else "offCharacter"


# --------------------------------------------------------------------------------------------------
# Aggregation.
# --------------------------------------------------------------------------------------------------


def _fight_deck_ids(first_record: Dict[str, Any]) -> List[str]:
    """The full deck ids from a fight's first record.

    In an isolated combat the run-level ``players[0].deck`` is empty — the built deck lives in the
    combat piles — so fall back to the union of hand + draw + discard + exhaust (at turn 1 that is the
    whole deck)."""
    state = first_record.get("state") or {}
    players = state.get("players") or []
    if not players:
        return []
    p0 = players[0]
    deck = p0.get("deck") or []
    ids = [c.get("cardId") for c in deck if c.get("cardId")]
    if ids:
        return ids
    combat = p0.get("combatState") or {}
    for pile in ("hand", "drawPile", "discardPile", "exhaustPile"):
        ids.extend(c.get("cardId") for c in (combat.get(pile) or []) if c.get("cardId"))
    return ids


def collect_fights(root: str) -> List[Dict[str, Any]]:
    """Reduce a corpus to one summary dict per fight (grouped by seed), tagging the on-disk split.

    Iterating per split lets us record where each fight was actually filed, so the determinism check
    can compare it against :func:`corpus.split_for_seed` (a real leak/stability test, not a tautology).
    """
    fights: Dict[str, Dict[str, Any]] = {}
    for split in corpus.SPLITS:
        for rec in corpus.iter_records(root, split):
            seed = rec.get("seed")
            meta = rec.get("scenarioMeta") or {}
            f = fights.get(seed)
            if f is None:
                f = {
                    "seed": seed, "onDiskSplit": split, "regime": meta.get("regime"),
                    "policy": meta.get("policy"), "act": meta.get("act"), "room": meta.get("room"),
                    "character": meta.get("character"), "deckSpec": meta.get("deckSpec"),
                    "removedCards": meta.get("removedCards"), "addedCards": meta.get("addedCards"),
                    "deckIds": _fight_deck_ids(rec), "records": 0, "won": False, "hpLost": 0.0,
                }
                fights[seed] = f
            f["records"] += 1
            if rec.get("done"):
                info = rec.get("info") or {}
                if info.get("won") is not None:
                    f["won"] = bool(info.get("won"))
                if info.get("hpLost") is not None:
                    f["hpLost"] = float(info.get("hpLost") or 0)
    return list(fights.values())


def _breakdown(fights: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Per-value fight/record counts + win-rate for one dimension, sorted by value."""
    groups: Dict[Any, Dict[str, Any]] = {}
    for f in fights:
        v = f.get(key)
        g = groups.setdefault(str(v), {"value": v, "fights": 0, "records": 0, "wins": 0})
        g["fights"] += 1
        g["records"] += f["records"]
        g["wins"] += 1 if f["won"] else 0
    rows = []
    for g in groups.values():
        g["winRate"] = g["wins"] / g["fights"] if g["fights"] else 0.0
        rows.append(g)
    return sorted(rows, key=lambda r: str(r["value"]))


def realistic_stats(fights: List[Dict[str, Any]],
                    card_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Removal/addition histograms, added-card pool distribution, and top-20 additions over the
    realistic-regime fights."""
    realistic = [f for f in fights if f.get("deckSpec") == "realistic"]
    removals_hist: Counter = Counter()
    additions_hist: Counter = Counter()
    pool_counts: Counter = Counter()
    added_counter: Counter = Counter()
    for f in realistic:
        removed = f.get("removedCards") or []
        added = f.get("addedCards") or []
        removals_hist[len(removed)] += 1
        additions_hist[len(added)] += 1
        for cid in added:
            added_counter[cid] += 1
            pool_counts[classify_added_card(cid, f.get("character"), card_meta)] += 1

    # Realized fractions over the four *configured* buckets (exclude unknown/other/ownOrOff).
    graded = {k: pool_counts.get(k, 0) for k in CONFIGURED_WEIGHTS}
    total_graded = sum(graded.values())
    realized = {k: (graded[k] / total_graded if total_graded else 0.0) for k in CONFIGURED_WEIGHTS}
    return {
        "fights": len(realistic),
        "removalsHist": dict(sorted(removals_hist.items())),
        "additionsHist": dict(sorted(additions_hist.items())),
        "poolCounts": dict(pool_counts),
        "poolRealized": realized,
        "poolConfigured": CONFIGURED_WEIGHTS,
        "totalGradedAdditions": total_graded,
        "top20Added": added_counter.most_common(20),
    }


def sample_realistic_decks(fights: List[Dict[str, Any]], n: int = 20,
                           rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    """Up to ``n`` realistic fights' (character, final deck ids) — deterministic sample by seed."""
    realistic = sorted((f for f in fights if f.get("deckSpec") == "realistic"),
                       key=lambda f: f["seed"])
    if len(realistic) > n:
        rng = rng or random.Random("corpus-report-sample")
        realistic = sorted(rng.sample(realistic, n), key=lambda f: f["seed"])
    return [{"seed": f["seed"], "character": f.get("character"),
             "removed": f.get("removedCards"), "added": f.get("addedCards"),
             "deck": sorted(f.get("deckIds") or [])} for f in realistic]


def determinism_check(fights: List[Dict[str, Any]], n: int = 3,
                      rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    """Re-derive ``n`` fights' splits from their seeds and confirm they match the on-disk split."""
    rng = rng or random.Random("corpus-report-determinism")
    picks = fights if len(fights) <= n else rng.sample(fights, n)
    out = []
    for f in picks:
        rederived = corpus.split_for_seed(f["seed"])
        out.append({
            "seed": f["seed"], "bucket": corpus.split_bucket(f["seed"]),
            "onDiskSplit": f["onDiskSplit"], "rederivedSplit": rederived,
            "stable": rederived == f["onDiskSplit"],
        })
    return out


def build_report(root: str, card_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Full machine-readable aggregation over the corpus at ``root``."""
    if card_meta is None:
        card_meta = load_card_meta()
    fights = collect_fights(root)
    total_records = sum(f["records"] for f in fights)
    return {
        "corpus": root,
        "totals": {"fights": len(fights), "records": total_records,
                   "cardMetaLoaded": bool(card_meta)},
        "composition": {key: _breakdown(fights, key)
                        for key in ("onDiskSplit", "regime", "policy", "act", "room", "character")},
        "realistic": realistic_stats(fights, card_meta),
        "sampleDecks": sample_realistic_decks(fights),
        "determinism": determinism_check(fights),
    }


# --------------------------------------------------------------------------------------------------
# Text rendering.
# --------------------------------------------------------------------------------------------------


def _fmt_breakdown(title: str, rows: List[Dict[str, Any]]) -> List[str]:
    out = [f"  {title:<12} {'fights':>8} {'records':>9} {'win%':>7}"]
    for r in rows:
        out.append(f"    {str(r['value']):<10} {r['fights']:>8} {r['records']:>9} "
                   f"{100 * r['winRate']:>6.1f}%")
    return out


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    t = report["totals"]
    lines.append("=" * 78)
    lines.append(f"CORPUS REPORT  {report['corpus']}")
    lines.append("=" * 78)
    lines.append(f"fights={t['fights']}  records={t['records']}  "
                 f"cards.json={'loaded' if t['cardMetaLoaded'] else 'ABSENT (pool stats degraded)'}")
    lines.append("")
    lines.append("-- Composition (fight win-rate) " + "-" * 46)
    labels = {"onDiskSplit": "split", "regime": "regime", "policy": "policy",
              "act": "act", "room": "room", "character": "character"}
    for key, rows in report["composition"].items():
        lines.extend(_fmt_breakdown(labels.get(key, key), rows))
        lines.append("")

    r = report["realistic"]
    lines.append("-- Realistic decks " + "-" * 59)
    lines.append(f"realistic fights={r['fights']}")
    lines.append(f"  #removals histogram: {r['removalsHist']}")
    lines.append(f"  #additions histogram: {r['additionsHist']}")
    lines.append("  added-card pool distribution (realized vs configured):")
    lines.append(f"    {'pool':<14}{'realized':>10}{'configured':>12}{'count':>8}")
    for k in CONFIGURED_WEIGHTS:
        lines.append(f"    {k:<14}{100 * r['poolRealized'][k]:>9.1f}%"
                     f"{100 * r['poolConfigured'][k]:>11.0f}%{r['poolCounts'].get(k, 0):>8}")
    extra = {k: v for k, v in r["poolCounts"].items() if k not in CONFIGURED_WEIGHTS}
    if extra:
        lines.append(f"    (ungraded: {extra})")
    lines.append(f"  total graded additions: {r['totalGradedAdditions']}")
    lines.append("  top-20 most-added cards:")
    for cid, cnt in r["top20Added"]:
        lines.append(f"    {cnt:>5}  {cid}")
    lines.append("")

    lines.append("-- Sample of realistic decks " + "-" * 49)
    for d in report["sampleDecks"]:
        lines.append(f"  {d['seed']}  [{d['character']}]  "
                     f"(-{len(d['removed'] or [])}/+{len(d['added'] or [])})")
        lines.append(f"      {d['deck']}")
    lines.append("")

    lines.append("-- Determinism (split re-derived from seed) " + "-" * 34)
    for c in report["determinism"]:
        flag = "OK" if c["stable"] else "*** MISMATCH ***"
        lines.append(f"  {c['seed']}  bucket={c['bucket']:>2}  onDisk={c['onDiskSplit']:<5} "
                     f"rederived={c['rederivedSplit']:<5} {flag}")
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lts2_agent.corpus_report",
                                description=__doc__.split("\n")[0])
    p.add_argument("--corpus", required=True, help="corpus root dir (as written by lts2_agent.collect)")
    p.add_argument("--cards", default=None, help="path to cards.json (default: packaged data/cards.json)")
    p.add_argument("--json", action="store_true", help="emit the aggregation as JSON instead of text")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.corpus, card_meta=load_card_meta(args.cards))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
