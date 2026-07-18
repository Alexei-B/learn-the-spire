"""Conditional-reachability table builder for the reachability-shaped synthetic experts (roadmap M3.5).

WHY this exists
---------------
The factored experts (cards / creatures / relics / potions / orbs) are trained on SYNTHETIC batches
generated in tokenizer-array space (:mod:`lts2_agent.wm.synth`). The design doctrine is that a synthetic
space must be **reachability-shaped**: identities are drawn from the real game vocabulary, and every
value a real state derives from that identity (a card's type/rarity/cost bounds, a creature's kind, a
power's amount range) is sampled from the identity's OWN observed conditional distribution — never
independently-uniform over the whole padded range. Only a thin uniform "wildcard" tail is kept as
coverage insurance for unseen ids.

The orbs generator (``synth.ORB_TYPES`` + ``_fill_orbs``) already does this by hand (5 orb types with
per-type value ranges). Cards and creatures have far too many identities to hand-author, so this module
MEASURES the conditional tables from the corpus once and freezes them into a JSON artifact that
``synth`` loads. Concretely it records, per tokenizer identity index:

* cards  (per ``cardIndex``):    observed type/rarity/targetType, observed enchant/afflict value
                                 frequencies, observed keyword multi-hot patterns, and per dynamic
                                 numeric column (``CARD_NUM`` minus the per-zone count vector) the
                                 observed ``(lo, hi)`` integer range.
* creatures (per ``identity``):  observed ``kind`` values + per ``CREATURE_NUM`` column ``(lo, hi)``.
* powers  (per ``powerIndex``):  ``amount`` ``(lo, hi)``.
* intents (per intent ``type``): per ``INTENT_NUM`` column ``(lo, hi)``.
* count histograms:              creatures-per-state, powers-per-state, intents-per-state, and
                                 powers-per-creature — these REPLACE the old absurd uniform
                                 ``0..MAX_POWERS`` / ``1..MAX_CREATURES`` count sampling.

Everything is keyed by the SAME integer indices the tokenizer emits: we build the tokenizer's canonical
view (:func:`tokens._canonical_from_state`) and read the already-mapped indices straight off it, so the
table never re-implements a hash or an enum ordering.

It is CPU-only, stdlib + numpy, and streams the corpus shard-strided exactly like :mod:`wm.ranges`
(reading whole shards spread across the corpus is the I/O-cheap way to span its breadth). A few hundred
thousand states is plenty to see every reachable card/creature and its value envelope.

CLI::

    python -m lts2_agent.wm.reachable --corpus data/corpus2 --out data/reachable_v1.json [--n N]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .. import corpus, tokens

# Numeric columns to record per type. Cards: the DYNAMIC content numerics (CARD_NUM minus the per-zone
# count vector, which synth fills from its own game-like small-int distribution). The others: their full
# numeric block (flag columns included — a flag's observed [lo,hi] is 0/1 and synth samples it directly).
_CARD_NUM_COLS = [c for c in tokens.CARD_NUM if c not in tokens.ZONE_COUNT_FIELDS]
_CREATURE_NUM_COLS = list(tokens.CREATURE_NUM)
_INTENT_NUM_COLS = list(tokens.INTENT_NUM)


def _iter_shard_strided(root: str, split: Optional[str], shard_stride: int):
    """Yield records from every ``shard_stride``-th shard (spread across the corpus). Whole-shard
    striding spans the corpus breadth (different acts/regimes land in different shards) while reading
    only ``1/shard_stride`` of the gzip bytes — the same I/O-cheap sampler :mod:`wm.ranges` uses."""
    paths = corpus.shard_paths(root, split)
    picked = paths[::shard_stride] if shard_stride > 1 else paths
    for p in picked:
        for rec in corpus.iter_shard(p):
            yield rec


def _upd_range(d: Dict[str, List[int]], col: str, v: int) -> None:
    """Track a running ``[min, max]`` for one column in an identity's range dict."""
    cur = d.get(col)
    if cur is None:
        d[col] = [v, v]
    else:
        if v < cur[0]:
            cur[0] = v
        if v > cur[1]:
            cur[1] = v


def _card_entry() -> Dict[str, Any]:
    return {
        "type": set(), "rarity": set(), "targetType": set(),
        "enchant": {}, "afflict": {},      # value -> observed frequency
        "keywords": set(),                 # set of on-bucket tuples (deduped multi-hot patterns)
        "num": {},                         # col -> [lo, hi]
    }


def _pick_root(root: str) -> str:
    """Use ``root`` if it has shards; otherwise fall back to ``data/corpus`` (a fresh clone / an older
    corpus that lacks the multi-act ``corpus2`` fields still yields a usable table)."""
    if corpus.shard_paths(root):
        return root
    alt = os.path.join(os.path.dirname(os.path.normpath(root)), "corpus")
    if alt != root and corpus.shard_paths(alt):
        return alt
    return root


def scan(root: str, split: Optional[str], n: Optional[int], shard_stride: int
         ) -> Tuple[Dict[str, Any], int]:
    """Stream shard-strided states and accumulate the conditional tables + count histograms. Returns
    ``(tables, n_states)`` where ``tables`` holds the still-pythonic accumulators."""
    cards: Dict[int, Dict[str, Any]] = {}
    creatures: Dict[int, Dict[str, Any]] = {}
    powers: Dict[int, List[int]] = {}
    intents: Dict[int, Dict[str, List[int]]] = {}
    counts = {
        "creatures_per_state": {}, "powers_per_state": {},
        "intents_per_state": {}, "powers_per_creature": {},
    }

    def _bump(hist: Dict[int, int], k: int) -> None:
        hist[k] = hist.get(k, 0) + 1

    n_states = 0
    src = (_iter_shard_strided(root, split, shard_stride) if shard_stride > 1
           else corpus.iter_records(root, split))
    for rec in src:
        for which in ("state", "nextState"):
            st = rec.get(which)
            if not st:
                continue
            cv = tokens._canonical_from_state(st)

            # Cards: population rows carry the tokenizer's mapped categorical indices + content numerics.
            for row in tokens._group_cards(cv):
                ci = int(row["cardIndex"])
                e = cards.get(ci)
                if e is None:
                    e = cards[ci] = _card_entry()
                e["type"].add(int(row["type"]))
                e["rarity"].add(int(row["rarity"]))
                e["targetType"].add(int(row["targetType"]))
                ench = int(row["enchant"])
                e["enchant"][ench] = e["enchant"].get(ench, 0) + 1
                affl = int(row["afflict"])
                e["afflict"][affl] = e["afflict"].get(affl, 0) + 1
                e["keywords"].add(tuple(int(b) for b in row["keywords"]))
                for col in _CARD_NUM_COLS:
                    _upd_range(e["num"], col, int(row[col]))

            # Creatures + their folded powers / intents.
            cr_list = cv["creatures"]
            _bump(counts["creatures_per_state"], len(cr_list))
            total_pw = total_in = 0
            for cr in cr_list:
                ident = int(cr["identity"])
                ce = creatures.get(ident)
                if ce is None:
                    ce = creatures[ident] = {"kind": set(), "num": {}}
                ce["kind"].add(int(cr["kind"]))
                for col in _CREATURE_NUM_COLS:
                    _upd_range(ce["num"], col, int(cr[col]))
                npw = len(cr["powers"])
                _bump(counts["powers_per_creature"], npw)
                total_pw += npw
                for pw in cr["powers"]:
                    pi = int(pw["idx"])
                    amt = int(pw["amount"])
                    cur = powers.get(pi)
                    if cur is None:
                        powers[pi] = [amt, amt]
                    else:
                        if amt < cur[0]:
                            cur[0] = amt
                        if amt > cur[1]:
                            cur[1] = amt
                for it in cr["intents"]:
                    total_in += 1
                    ty = int(it["type"])
                    ie = intents.get(ty)
                    if ie is None:
                        ie = intents[ty] = {}
                    for col in _INTENT_NUM_COLS:
                        _upd_range(ie, col, int(it[col]))
            _bump(counts["powers_per_state"], total_pw)
            _bump(counts["intents_per_state"], total_in)

            n_states += 1
            if n and n_states >= n:
                return {"cards": cards, "creatures": creatures, "powers": powers,
                        "intents": intents, "counts": counts}, n_states
    return {"cards": cards, "creatures": creatures, "powers": powers,
            "intents": intents, "counts": counts}, n_states


def _jsonable(tables: Dict[str, Any], root: str, n_states: int) -> Dict[str, Any]:
    """Convert the pythonic accumulators into the on-disk JSON shape (string keys, sorted lists). All
    identity keys stay the tokenizer's integer index, stringified only because JSON object keys must be
    strings — :func:`synth._parse_reachable` casts them straight back to int."""
    cards_out: Dict[str, Any] = {}
    for ci, e in tables["cards"].items():
        cards_out[str(ci)] = {
            "type": sorted(e["type"]),
            "rarity": sorted(e["rarity"]),
            "targetType": sorted(e["targetType"]),
            "enchant": {str(k): v for k, v in sorted(e["enchant"].items())},
            "afflict": {str(k): v for k, v in sorted(e["afflict"].items())},
            "keywords": sorted([list(t) for t in e["keywords"]]),
            "num": {col: [lo, hi] for col, (lo, hi) in
                    ((c, e["num"][c]) for c in _CARD_NUM_COLS if c in e["num"])},
        }
    creatures_out: Dict[str, Any] = {}
    for ident, ce in tables["creatures"].items():
        creatures_out[str(ident)] = {
            "kind": sorted(ce["kind"]),
            "num": {col: [lo, hi] for col, (lo, hi) in
                    ((c, ce["num"][c]) for c in _CREATURE_NUM_COLS if c in ce["num"])},
        }
    powers_out = {str(pi): [lo, hi] for pi, (lo, hi) in sorted(tables["powers"].items())}
    intents_out = {
        str(ty): {col: ie[col] for col in _INTENT_NUM_COLS if col in ie}
        for ty, ie in sorted(tables["intents"].items())
    }
    counts_out = {name: {str(k): v for k, v in sorted(hist.items())}
                  for name, hist in tables["counts"].items()}
    return {
        "meta": {
            "tokenizer_signature": tokens.tokenizer_signature(),
            "root": root, "states": n_states,
        },
        "cards": cards_out,
        "creatures": creatures_out,
        "powers": powers_out,
        "intents": intents_out,
        "counts": counts_out,
    }


# Estimated states per shard (state + nextState per record, ~2000 records/shard, some done-nextState
# nulls) — used only to auto-pick a shard stride that reads ~n states from EVENLY-SPREAD shards.
_EST_STATES_PER_SHARD = 3800


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the conditional-reachability table for the synthetic experts.")
    ap.add_argument("--corpus", default="data/corpus2", help="corpus root to stream (default data/corpus2)")
    ap.add_argument("--out", default="data/reachable_v1.json", help="output JSON artifact path")
    ap.add_argument("--split", default=None, help="restrict to one split (default: all splits)")
    ap.add_argument("--n", type=int, default=400000, help="max states to scan (default 400k)")
    ap.add_argument("--shard-stride", type=int, default=0,
                    help="read every N-th shard (0 = auto: spread ~n states across the corpus)")
    args = ap.parse_args(argv)

    root = _pick_root(args.corpus)
    paths = corpus.shard_paths(root, args.split)
    if not paths:
        print("no shards under %s" % root)
        return 1
    stride = args.shard_stride
    if stride <= 0:
        # Pick a stride so the evenly-spread picked shards hold ~n states (no leading-shard bias).
        stride = max(1, round(len(paths) * _EST_STATES_PER_SHARD / max(1, args.n)))

    tables, n_states = scan(root, args.split, args.n or None, stride)
    doc = _jsonable(tables, root, n_states)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))

    print("=" * 78)
    print("REACHABLE TABLE  (%s)" % tokens.tokenizer_signature())
    print("root %s   split %s   shard-stride %d   states scanned %d"
          % (root, args.split or "ALL", stride, n_states))
    print("-" * 78)
    print("  cards:     %5d observed cardIndex identities" % len(doc["cards"]))
    print("  creatures: %5d observed identity indices" % len(doc["creatures"]))
    print("  powers:    %5d observed powerIndex identities" % len(doc["powers"]))
    print("  intents:   %5d observed intent types" % len(doc["intents"]))
    for name, hist in doc["counts"].items():
        ks = sorted(int(k) for k in hist)
        print("  count[%-20s] range %d..%d over %d distinct values"
              % (name, ks[0] if ks else 0, ks[-1] if ks else 0, len(ks)))
    print("  wrote %s" % args.out)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
