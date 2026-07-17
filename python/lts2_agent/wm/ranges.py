"""Per-field integer range scan — the v3 exactness contract's measurement tool (roadmap M3.5).

Tokenizer v3 gives every numeric column a measured integer ``(lo, hi, resolution)`` range (stored in
:data:`lts2_agent.wm.spec.NUMERIC_RANGES`) so a future per-field decoder can bin each quantity exactly
instead of regressing one shared symlog float. This module streams a corpus, builds the tokenizer's
canonical view of every state (:func:`lts2_agent.tokens._canonical_from_state`), and records the
observed min/max of each numeric field per token type — then prints a ``_RANGES_RAW`` dict literal
(with slack on ``hi``) to paste into the spec.

It mirrors :mod:`lts2_agent.wm.footprint`: CPU-only, corpus-streaming, stdlib + numpy. Booleans /
presence flags (``costsX``, ``hasDamage``, ``active``, …) are 0..1 by construction and are excluded.

CLI::

    python -m lts2_agent.wm.ranges --corpus data/corpus            # whole corpus (all splits)
    python -m lts2_agent.wm.ranges --corpus data/corpus --n 200000 # first 200k states
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Dict, Optional, Tuple

from .. import corpus, tokens

# Numeric (non-boolean) columns to scan, per token type. Names match the tokenizer's canonical dicts.
_BOOL_CARD = {"costsX", "upgraded", "canPlay", "hasDamage", "hasBlock", "hasSummon"}
_CARD_SCALARS = [c for c in tokens.CARD_NUM if c not in _BOOL_CARD and c not in tokens.ZONE_COUNT_FIELDS]
_INTENT_BOOL = {"hasDamage", "hasHits"}
_INTENT_SCALARS = [c for c in tokens.INTENT_NUM if c not in _INTENT_BOOL]


def _upd(acc: Dict[str, Dict[str, list]], tname: str, col: str, v: int) -> None:
    d = acc.setdefault(tname, {})
    cur = d.get(col)
    if cur is None:
        d[col] = [v, v]
    else:
        if v < cur[0]:
            cur[0] = v
        if v > cur[1]:
            cur[1] = v


_MAX_KEYS = ("cards", "creatures", "powers", "intents", "orbs", "relics", "potions")


def _iter_shard_strided(root: str, split: Optional[str], shard_stride: int):
    """Yield records from every ``shard_stride``-th shard (spread across the corpus). Sampling whole
    shards — not records within all shards — spans the corpus breadth (different acts/regimes land in
    different shards) while reading only ``1/shard_stride`` of the gzip bytes, so it is I/O-cheap."""
    paths = corpus.shard_paths(root, split)
    picked = paths[::shard_stride] if shard_stride > 1 else paths
    for p in picked:
        for rec in corpus.iter_shard(p):
            yield rec


def scan(root: str, split: Optional[str], n: Optional[int], stride: int = 1, shard_stride: int = 1
         ) -> Tuple[Dict[str, Dict[str, list]], Dict[str, int], Dict[str, float], int]:
    """Stream states (``stride`` = keep 1-of-N records; ``shard_stride`` = read only every N-th shard,
    the I/O-cheap way to span the whole corpus breadth instead of over-sampling the homogeneous leading
    shards), accumulating per-field ``{type: {col: [min, max]}}`` ranges, per-type token maxima,
    population sums, and the state count."""
    acc: Dict[str, Dict[str, list]] = {}
    maxima: Dict[str, int] = {k: 0 for k in _MAX_KEYS}
    sums = {"instances": 0.0, "rows": 0.0}
    n_states = 0
    ridx = -1
    src = (_iter_shard_strided(root, split, shard_stride) if shard_stride > 1
           else corpus.iter_records(root, split))
    for rec in src:
        ridx += 1
        if stride > 1 and (ridx % stride):
            continue
        for which in ("state", "nextState"):
            st = rec.get(which)
            if not st:
                continue
            cv = tokens._canonical_from_state(st)

            g = cv["global"]
            for col in tokens.GLOBAL_NUM:
                _upd(acc, "global", col, int(g[col]))
            pend = cv.get("pending")
            if pend is not None:
                _upd(acc, "pending", "minSelect", int(pend["minSelect"]))
                _upd(acc, "pending", "maxSelect", int(pend["maxSelect"]))

            # Card scalars over per-instance dicts; per-zone counts over the grouped population rows.
            for z in tokens.ZONES:
                for c in cv["cards"][z]:
                    for col in _CARD_SCALARS:
                        _upd(acc, "card", col, int(c[col]))
            rows = tokens._group_cards(cv)
            for row in rows:
                for z in tokens.ZONES:
                    _upd(acc, "card", "count_" + z, int(row["counts"][z]))

            for cr in cv["creatures"]:
                for col in ("currentHp", "maxHp", "block", "combatId"):
                    _upd(acc, "creature", col, int(cr[col]))
                for pw in cr["powers"]:
                    _upd(acc, "power", "amount", int(pw["amount"]))
                for it in cr["intents"]:
                    for col in _INTENT_SCALARS:
                        _upd(acc, "intent", col, int(it[col]))
            for orb in cv["orbs"]:
                _upd(acc, "orb", "passiveValue", int(orb["passiveValue"]))
                _upd(acc, "orb", "evokeValue", int(orb["evokeValue"]))

            instances = sum(len(cv["cards"][z]) for z in tokens.ZONES)
            maxima["cards"] = max(maxima["cards"], len(rows))
            maxima["creatures"] = max(maxima["creatures"], len(cv["creatures"]))
            maxima["powers"] = max(maxima["powers"], sum(len(c["powers"]) for c in cv["creatures"]))
            maxima["intents"] = max(maxima["intents"], sum(len(c["intents"]) for c in cv["creatures"]))
            maxima["orbs"] = max(maxima["orbs"], len(cv["orbs"]))
            maxima["relics"] = max(maxima["relics"], len(cv["relics"]))
            maxima["potions"] = max(maxima["potions"], len(cv["potions"]))
            sums["instances"] += instances
            sums["rows"] += len(rows)
            n_states += 1
            if n and n_states >= n:
                return acc, maxima, sums, n_states
    return acc, maxima, sums, n_states


def _slack_hi(hi: int) -> int:
    """Generous upper slack so real play never clamps: >=25% headroom (min +5), rounded up to a
    readable step. The tokenizer's NUM_CLIP sentinel (999999999 -> clamp) passes through unchanged."""
    if hi >= tokens.NUM_CLIP:
        return tokens.NUM_CLIP
    if hi <= 0:
        return max(hi, 1)
    padded = hi + max(5, math.ceil(hi * 0.25))
    if padded <= 100:
        step = 10
    elif padded <= 1000:
        step = 50
    elif padded <= 10000:
        step = 500
    else:
        step = 5000
    return int(math.ceil(padded / step) * step)


# Token-type print order.
_ORDER = ["global", "pending", "card", "creature", "power", "intent", "orb"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scan per-field integer ranges for the v3 exactness spec.")
    ap.add_argument("--corpus", default="data/corpus", help="corpus root to stream")
    ap.add_argument("--split", default=None, help="restrict to one split (default: all splits)")
    ap.add_argument("--n", type=int, default=None, help="max states to scan (0/None = all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="scan 1 of every STRIDE records (default 1)")
    ap.add_argument("--shard-stride", type=int, default=1,
                    help="read only every N-th shard (I/O-cheap breadth sampling; default 1 = all)")
    args = ap.parse_args(argv)

    acc, maxima, sums, n_states = scan(args.corpus, args.split, args.n or None,
                                       max(1, args.stride), max(1, args.shard_stride))
    if not acc:
        print("no states scanned")
        return 1

    print("=" * 78)
    print(f"PER-FIELD INTEGER RANGES   ({tokens.tokenizer_signature()})")
    print(f"root {args.corpus}   split {args.split or 'ALL'}   stride {args.stride}   "
          f"states {n_states}")
    print("=" * 78)
    caps = {"cards": tokens.MAX_CARDS, "creatures": tokens.MAX_CREATURES, "powers": tokens.MAX_POWERS,
            "intents": tokens.MAX_INTENTS, "orbs": tokens.MAX_ORBS, "relics": tokens.MAX_RELICS,
            "potions": tokens.MAX_POTIONS}
    print("token maxima (this scan) vs padded caps:")
    for k in _MAX_KEYS:
        flag = "  !! OVER CAP" if maxima[k] > caps[k] else ""
        print(f"  {k:10s} max {maxima[k]:4d}   cap {caps[k]:4d}{flag}")
    if n_states:
        print(f"  card population rows/state — mean instances {sums['instances'] / n_states:6.2f}  ->  "
              f"mean rows {sums['rows'] / n_states:6.2f}  "
              f"({sums['instances'] / max(1.0, sums['rows']):.2f}x shorter)")
    print("=" * 78)
    print("_RANGES_RAW = {")
    for tname in _ORDER:
        if tname not in acc:
            continue
        print(f'    "{tname}": {{')
        for col in sorted(acc[tname]):
            lo, hi = acc[tname][col]
            hi_s = _slack_hi(hi)
            print(f'        "{col}": ({max(0, lo) if lo >= 0 else lo}, {hi_s}, 1),'
                  f'   # observed [{lo}, {hi}]')
        print("    },")
    print("}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
