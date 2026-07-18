"""Synthetic-space batch generators for the factored experts (roadmap M3.5 synthetic-space training).

The product decision: a **finite-space** expert (potions / relics / orbs, and — as coverage insurance —
creatures / cards) is decoupled from game data and trained on **synthetic uniform configurations
generated mechanically in tokenizer-array space**. The decoder is the predictor's API and must decode
ANY valid array configuration, not just the game-frequent ones; uniform coverage kills the rare-tail
floors that game-frequency training leaves behind (measured: potions capped at 0.995 by ~3 rare belt
configs — non-left-packed belts, rare combos — and relics starved by frequency imbalance).

Each generator emits a **ready model-input batch**: the full :data:`model.BATCH_KEYS` array set, stacked
``[B, ...]``, with the *target* expert(s)' keys sampled uniformly-with-design over their space and every
other key left in its trivial empty encoding (mask all-False, ids 0, numerics 0). A solo/mixed run only
ever reads the trained expert's slice, so the empty remainder is inert; keeping the full key set means a
synthetic batch is byte-shape-identical to a real cache batch and the two concatenate for ``mixed:R``.

Conventions preserved EXACTLY (mirrors :func:`tokens.tokenize`):

* **Presence is left-packed**: the first ``k`` slots of a variable-length type are present (mask True),
  the rest padded (mask False, arrays 0). ``k`` is the sampled item count.
* **index-0 semantics**: a potion slot id 0 is an EMPTY belt slot (a present token); a catalog id 0 is
  none/unknown. Relic ids are drawn 1..N-1 (a present relic is a real relic); duplicates are LEGAL and
  drawn on purpose (rare in-game, e.g. via wax relics / events) and relics are POSITIONAL (v5) — one row
  per instance carrying an explicit ``slot`` == its list index (acquisition order is semantic). Enum
  columns exclude the reserved trailing UNKNOWN slot.
* **symlog storage**: every non-flag numeric column stores ``tokens.symlog(int)`` (flags store the raw
  0/1), so :meth:`experts.RangeBinHeads.bin_targets` recovers the exact integer — the same integer<->
  stored mapping the tokenizer uses. Values are sampled uniformly inside :data:`spec.NUMERIC_RANGES`.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .. import tokens
from . import model as M
from . import spec as S
from .experts import EXPERT_TYPES, RAW_NUM_COLS

# Token types whose leading categorical column is a small FIXED ENUM (vocab == len(table)+1, the trailing
# slot reserved for UNKNOWN — never emitted by a real state). Sampling excludes that reserved slot.
_ENUM_TOP_RESERVED = {
    ("global", "phase"), ("global", "side"), ("global", "turnPhase"),
    ("card", "type"), ("card", "rarity"), ("card", "targetType"),
    ("creature", "kind"), ("intent", "type"),
}


# Potion belt is a hard physical game limit (corpus max 5 slots) — cover 5 + a small margin, NOT the
# MAX_POTIONS=8 padded dim (a >6-slot belt is not a valid configuration). Every other cap is the token
# type's padded dim (relics accumulate to MAX_RELICS; orbs to MAX_ORBS): those are the real game ranges.
POTION_MAX_BELT = min(6, tokens.MAX_POTIONS)


def _symlog_arr(a: np.ndarray) -> np.ndarray:
    """Array-wise symlog — identical formula to :func:`tokens.symlog` (which is scalar-only: it casts to a
    python float). Kept as the vectorized twin so a whole numeric block encodes at once; a test asserts it
    agrees with ``tokens.symlog`` element-for-element so it can never drift from the tokenizer's storage."""
    return np.sign(a) * np.log1p(np.abs(a))


def _symexp_arr(a: np.ndarray) -> np.ndarray:
    """Array-wise inverse of :func:`_symlog_arr` (== :func:`tokens.symexp` element-wise): recover a stored
    numeric's real value from symlog storage. Used only to decode a canonical-sort key numeric back to its
    integer when ordering generated intent rows the way the tokenizer flattens them."""
    return np.sign(a) * np.expm1(np.abs(a))


# ==================================================================================================
# Conditional-reachability table (data/reachable_v3.json, built by wm.reachable). The cards + creatures
# generators are REACHABILITY-SHAPED: identities are drawn from the corpus-observed set, and every value
# a real state derives from an identity (a card's type/rarity/cost envelope, a creature's kind, a power's
# amount range) is sampled from THAT identity's observed conditional range (+ a 1.5x margin, clamped to
# the spec range) instead of independently-uniform over the whole padded space — mirroring the orbs
# generator's ORB_TYPES design, but measured rather than hand-authored. A thin per-row WILDCARD tail keeps
# the old fully-uniform path as coverage insurance for unseen ids. The table is loaded lazily once into a
# module cache; a test may inject a parsed fixture straight into ``_REACHABLE_TABLE`` (see _parse_reachable
# for the shape) so the suite runs without the (gitignored) artifact.
# ==================================================================================================

CARD_WILDCARD_PROB = 0.01        # per card row: STRUCTURED id-level insurance tail (v3: 0.05 -> 0.01). A
                                 # wildcard row draws a uniform cardIndex over the full vocab (coverage for
                                 # unseen ids) but injects NO incompressible bit noise — empty keywords,
                                 # enchant/afflict pinned to their real marginal mode (0), numerics per spec.
CREATURE_WILDCARD_PROB = 0.05    # per creature/power/intent row: same uniform fallback

# Large-deck (>=CARD_BIGDECK_THRESH instances) oversampling for the cards generator. TRAIN-TIME exposure of
# big decks is deliberately boosted ABOVE corpus frequency (they carry the bulk of real per-row error yet
# are rare in the instances-per-state histogram tail): with probability CARD_BIGDECK_BOOST a state's
# instance count is redrawn from the renormalized >=THRESH tail of that histogram. The boost is applied ONLY
# on the training streams (synth_batches / mixed_batches); coverage_val_sample keeps the UNBOOSTED,
# corpus-shaped distribution so the coverage yardstick stays comparable to real val.
CARD_BIGDECK_BOOST = 0.15
CARD_BIGDECK_THRESH = 16

_REACHABLE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "reachable_v4.json")   # v7: absolute-keyword-flag patterns (was reachable_v3, hashed added kw)

_REACHABLE_TABLE: Optional[Dict[str, Any]] = None   # module cache (tests may inject a parsed fixture)


def _norm_probs(counts: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(counts), dtype=np.float64)
    s = a.sum()
    return (a / s) if s > 0 else np.full(len(a), 1.0 / max(1, len(a)))


def _parse_kw_patterns(kw_field: Iterable[Any]) -> Tuple[List[Tuple[int, ...]], List[float]]:
    """Parse a card's on-disk ``keywords`` field into ``(patterns, counts)``. Accepts BOTH shapes:

    * v3 (frequency): a list of ``[on-bucket list, count]`` pairs — patterns sampled by observed frequency.
    * legacy (deduped list): a list of plain on-bucket lists — each assigned an equal count of 1 (so an old
      reachable_v2 artifact / a legacy test fixture still parses, just uniform over its patterns).

    An empty field yields the single empty pattern ``[()]`` at count 1 (a card with no observed keywords)."""
    pats: List[Tuple[int, ...]] = []
    cnts: List[float] = []
    for el in kw_field:
        if len(el) == 2 and isinstance(el[0], list) and isinstance(el[1], (int, float)):
            pat, c = el[0], float(el[1])          # v3 [pattern, count] pair
        else:
            pat, c = el, 1.0                       # legacy bare pattern list
        pats.append(tuple(int(b) for b in pat))
        cnts.append(c)
    if not pats:
        return [()], [1.0]
    return pats, cnts


def _parse_reachable(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the on-disk JSON (string keys, count dicts) into the parsed in-memory table the fillers
    sample from: integer identity keys, numpy id arrays, per-identity value sets / (value, prob) arrays,
    per-column ``(lo, hi)`` integer ranges, and per-count-histogram ``(values, probs)`` arrays."""
    cards: Dict[int, Dict[str, Any]] = {}
    for k, e in doc["cards"].items():
        ench = sorted((int(vk), vv) for vk, vv in e["enchant"].items())
        affl = sorted((int(vk), vv) for vk, vv in e["afflict"].items())
        kw_pats, kw_cnts = _parse_kw_patterns(e["keywords"])
        cards[int(k)] = {
            "type": np.asarray(e["type"], np.int64),
            "rarity": np.asarray(e["rarity"], np.int64),
            "targetType": np.asarray(e["targetType"], np.int64),
            "enchant_vals": np.asarray([v for v, _ in ench], np.int64),
            "enchant_p": _norm_probs(c for _, c in ench),
            "afflict_vals": np.asarray([v for v, _ in affl], np.int64),
            "afflict_p": _norm_probs(c for _, c in affl),
            "keywords": kw_pats,
            "keyword_p": _norm_probs(kw_cnts),
            "num": {col: (int(lo), int(hi)) for col, (lo, hi) in e["num"].items()},
        }
    creatures: Dict[int, Dict[str, Any]] = {}
    for k, ce in doc["creatures"].items():
        creatures[int(k)] = {
            "kind": np.asarray(ce["kind"], np.int64),
            "num": {col: (int(lo), int(hi)) for col, (lo, hi) in ce["num"].items()},
        }
    powers = {int(k): (int(v[0]), int(v[1])) for k, v in doc["powers"].items()}
    intents = {int(k): {col: (int(lo), int(hi)) for col, (lo, hi) in v.items()}
               for k, v in doc["intents"].items()}
    counts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, hist in doc["counts"].items():
        items = sorted((int(kk), vv) for kk, vv in hist.items())
        counts[name] = (np.asarray([kk for kk, _ in items], np.int64),
                        _norm_probs(vv for _, vv in items))
    tbl = {
        "cards": cards, "card_ids": np.asarray(sorted(cards), np.int64),
        "creatures": creatures, "creature_ids": np.asarray(sorted(creatures), np.int64),
        "powers": powers, "power_ids": np.asarray(sorted(powers), np.int64),
        "intents": intents, "intent_ids": np.asarray(sorted(intents), np.int64),
        "counts": counts,
    }
    _augment_reachable(tbl)
    return tbl


def _augment_reachable(tbl: Dict[str, Any]) -> None:
    """Precompute per-identity numeric ``(lo, hi)`` bound arrays (one row per identity, in the sorted id
    order of ``*_ids``) so the vectorized fillers draw a whole batch of table rows' numerics in ONE
    ``rng.integers`` call over gathered bounds — the reachability margin+clamp (:func:`_reach_lohi`) is
    baked in here, off the hot path, instead of being recomputed per row. This is the single change that
    lets ``_fill_cards`` / ``_fill_creature_stats`` avoid a per-row Python table lookup."""
    def _lohi_block(get_num, ids: np.ndarray, type_name: str, cols: List[str]):
        lo = np.zeros((len(ids), len(cols)), np.int64)
        hi = np.zeros((len(ids), len(cols)), np.int64)
        for r, ident in enumerate(ids.tolist()):
            num = get_num(ident)
            for j, col in enumerate(cols):
                clo, chi = num.get(col, (0, 0))
                lo[r, j], hi[r, j] = _reach_lohi(type_name, col, clo, chi)
        return lo, hi

    card_cols = [col for _, col, _ in _CARD_NONZONE_NUM]
    tbl["card_num_lo"], tbl["card_num_hi"] = _lohi_block(
        lambda i: tbl["cards"][i]["num"], tbl["card_ids"], "card", card_cols)
    tbl["creature_num_lo"], tbl["creature_num_hi"] = _lohi_block(
        lambda i: tbl["creatures"][i]["num"], tbl["creature_ids"], "creature", list(tokens.CREATURE_NUM))
    tbl["intent_num_lo"], tbl["intent_num_hi"] = _lohi_block(
        lambda i: tbl["intents"][i], tbl["intent_ids"], "intent", list(tokens.INTENT_NUM))
    pids = tbl["power_ids"]
    plo = np.zeros(len(pids), np.int64)
    phi = np.zeros(len(pids), np.int64)
    for r, pid in enumerate(pids.tolist()):
        lo, hi = tbl["powers"][pid]
        plo[r], phi[r] = _reach_lohi("power", "amount", lo, hi)
    tbl["power_amt_lo"], tbl["power_amt_hi"] = plo, phi

    # Full-vocab amount bound tables (one entry per powerIndex 0..vocab-1): an observed power keeps its
    # reachability (lo, hi); any other id (drawn only by the wildcard tail) gets the full spec range. Lets
    # the distinct-power sampler gather a whole batch of amount bounds by powerIndex in one indexing op.
    pw_vocab = S.TYPE_BY_NAME["power"].cat_cols[0][1]
    spec_amt = S.NUMERIC_RANGES["power"]["amount"]
    amt_lo_by_id = np.full(pw_vocab, spec_amt.lo, np.int64)
    amt_hi_by_id = np.full(pw_vocab, spec_amt.hi, np.int64)
    amt_lo_by_id[pids], amt_hi_by_id[pids] = plo, phi
    tbl["power_amt_lo_by_id"], tbl["power_amt_hi_by_id"] = amt_lo_by_id, amt_hi_by_id

    # Ragged->padded gather tables for the card's identity-conditioned categoricals, so a whole batch of
    # table rows draws them WITHOUT a per-identity Python loop: uniform columns (type/rarity/targetType)
    # store the option values + per-card option counts (draw = pick a random slot < count); weighted
    # columns (enchant/afflict) store the values + a per-card cumulative-prob row (draw = inverse-CDF,
    # ``argmax(cdf >= u)``); the keyword pattern set becomes a per-card [pattern, len(KEYWORDS)] multi-hot
    # tensor (draw = pick a random pattern row). All widths are small (enum sizes / distinct-value counts).
    cards = tbl["cards"]
    cids = tbl["card_ids"]

    def _pad_uniform(key: str) -> None:
        opts = [cards[i][key] for i in cids.tolist()]
        ln = np.array([len(o) for o in opts], np.int64)
        w = int(ln.max()) if len(ln) else 1
        mat = np.zeros((len(opts), max(1, w)), np.int64)
        for r, o in enumerate(opts):
            mat[r, :len(o)] = o
        tbl["card_" + key + "_vals"], tbl["card_" + key + "_len"] = mat, ln

    for k in ("type", "rarity", "targetType"):
        _pad_uniform(k)

    def _pad_weighted(vkey: str, pkey: str, out: str) -> None:
        vv = [cards[i][vkey] for i in cids.tolist()]
        pp = [cards[i][pkey] for i in cids.tolist()]
        w = max((len(v) for v in vv), default=1)
        vmat = np.zeros((len(vv), max(1, w)), np.int64)
        cdf = np.ones((len(vv), max(1, w)), np.float64)   # pad with 1.0 so a pad slot never wins argmax
        for r, (v, p) in enumerate(zip(vv, pp)):
            vmat[r, :len(v)] = v
            c = np.cumsum(p)
            if len(c):
                c[-1] = 1.0                               # guard fp so some real slot always satisfies u<1
            cdf[r, :len(c)] = c
        tbl["card_" + out + "_vals"], tbl["card_" + out + "_cdf"] = vmat, cdf

    _pad_weighted("enchant_vals", "enchant_p", "enchant")
    _pad_weighted("afflict_vals", "afflict_p", "afflict")

    # Keyword pattern set becomes a per-card [pattern, len(KEYWORDS)] multi-hot tensor PLUS a per-card
    # cumulative-frequency row (draw = inverse-CDF over the OBSERVED pattern frequencies — a card's canonical
    # ABSOLUTE pattern is emitted at its real rate and rare alternates stay rare). Widths are tiny (distinct
    # patterns). v7: patterns are ABSOLUTE keyword-flag index tuples (0..6), not the old hashed buckets.
    pats_per = [cards[i]["keywords"] for i in cids.tolist()]
    probs_per = [cards[i]["keyword_p"] for i in cids.tolist()]
    pmax = max((len(p) for p in pats_per), default=1)
    kw_mat = np.zeros((len(pats_per), max(1, pmax), len(tokens.KEYWORDS)), np.float32)
    kw_cdf = np.ones((len(pats_per), max(1, pmax)), np.float64)   # pad with 1.0 so a pad slot never wins argmax
    for r, (pats, p) in enumerate(zip(pats_per, probs_per)):
        for pi, pat in enumerate(pats):
            for bkt in pat:
                kw_mat[r, pi, bkt] = 1.0
        c = np.cumsum(p)
        if len(c):
            c[-1] = 1.0                                           # guard fp so some real pattern always wins
        kw_cdf[r, :len(c)] = c
    tbl["card_kw_mat"], tbl["card_kw_cdf"] = kw_mat, kw_cdf


def _load_reachable() -> Dict[str, Any]:
    """Return the parsed reachability table (cached). Raises a clear, actionable error if the artifact is
    missing — the cards/creatures generators MUST NOT silently fall back to the old uniform space."""
    global _REACHABLE_TABLE
    if _REACHABLE_TABLE is None:
        if not os.path.exists(_REACHABLE_JSON):
            raise FileNotFoundError(
                "reachability table not found at " + _REACHABLE_JSON + "; build it with:\n"
                "    python -m lts2_agent.wm.reachable --corpus data/corpus2 --out data/reachable_v4.json")
        with open(_REACHABLE_JSON, encoding="utf-8") as f:
            _REACHABLE_TABLE = _parse_reachable(json.load(f))
    return _REACHABLE_TABLE


def _try_load_reachable() -> Optional[Dict[str, Any]]:
    """Like :func:`_load_reachable` but returns ``None`` when the artifact is absent (used by the entropy
    estimators in :mod:`wm.convergence`, which keep an analytic no-table fallback)."""
    try:
        return _load_reachable()
    except FileNotFoundError:
        return None


def _margin_lohi(type_name: str, col: str, lo: int, hi: int) -> Tuple[int, int]:
    """The reachability margin rule: widen an observed integer range ~1.5x outward (lo'=floor(lo*1.5) when
    lo<0 else lo; hi'=ceil(hi*1.5) when hi>0 else hi), then clamp to the spec's decode range so a sampled
    value is always exactly representable. Never inverts the interval."""
    lo2 = math.floor(lo * 1.5) if lo < 0 else lo
    hi2 = math.ceil(hi * 1.5) if hi > 0 else hi
    lo2, _ = S.clamp_to_range(type_name, col, int(lo2))
    hi2, _ = S.clamp_to_range(type_name, col, int(hi2))
    if lo2 > hi2:
        lo2 = hi2
    return int(lo2), int(hi2)


def _reach_lohi(type_name: str, col: str, lo: int, hi: int) -> Tuple[int, int]:
    """Sampling bounds for a reachability numeric column: a dynamic numeric (has a spec range) gets the
    1.5x margin + clamp; a flag / no-range column keeps its observed integer range verbatim (so a 0/1
    flag stays a 0/1 flag rather than being widened to 2)."""
    if S.NUMERIC_RANGES.get(type_name, {}).get(col) is None:
        return int(lo), int(hi)
    return _margin_lohi(type_name, col, lo, hi)


def reach_bins(type_name: str, col: str, lo: int, hi: int) -> int:
    """Number of distinct integers the reshaped generator can emit for one identity's numeric column —
    the entropy estimators in :mod:`wm.convergence` turn this into ``log2(bins)``."""
    l, h = _reach_lohi(type_name, col, lo, hi)
    return h - l + 1


# ==================================================================================================
# v6: cards are INSTANCE rows. Their game-shaped marginals (instances-per-state count histogram + the
# per-zone instance marginal) now live in the reachability table (data/reachable_v3.json,
# ``counts["instances_per_state"]`` and ``counts["card_zone"]``), sampled exactly like the creature/power
# count histograms — so the old population-row constants (_CARD_ROWS_HIST / _CARD_NZONES_HIST /
# _CARD_ZONE_WEIGHTS / _CARD_ZONE_COUNT_HIST) and the zone-vector sampler are deleted. Only ids + dynamic
# numerics remain uniform-with-design (the per-cardIndex conditional table + a CARD_WILDCARD_PROB tail).
# ==================================================================================================


# ==================================================================================================
# Empty full batch template (all BATCH_KEYS at their padded shapes, trivial empty encoding).
# ==================================================================================================

def _zeros_batch(B: int) -> Dict[str, np.ndarray]:
    z: Dict[str, np.ndarray] = {
        "global_idx": np.zeros((B, 1, len(tokens.GLOBAL_IDX)), np.int32),
        "global_num": np.zeros((B, 1, len(tokens.GLOBAL_NUM)), np.float32),
        "pending": np.zeros((B, 1, len(tokens.PENDING_NUM)), np.float32),
        "card_kw": np.zeros((B, tokens.MAX_CARDS, len(tokens.KEYWORDS)), np.float32),
    }
    for t in S.VARIABLE_TYPES:
        z[t.idx_key] = np.zeros((B, t.max_slots, len(t.cat_cols)), np.int32)
        if t.num_key:
            z[t.num_key] = np.zeros((B, t.max_slots, t.num_width), np.float32)
        z[t.mask_key] = np.zeros((B, t.max_slots), np.bool_)
    # Sanity: the template must carry exactly the keys the model consumes.
    assert set(z) == set(M.BATCH_KEYS), (set(z) ^ set(M.BATCH_KEYS))
    return z


# ==================================================================================================
# Per-field sampling helpers.
# ==================================================================================================

def _cat_high(type_name: str, col_name: str, vocab: int) -> int:
    """Exclusive upper bound for a categorical column's sampled index: an enum column stops before its
    reserved UNKNOWN top slot; a catalog/hash column spans the whole vocab (index 0 == none/unknown is a
    legal id there)."""
    return (vocab - 1) if (type_name, col_name) in _ENUM_TOP_RESERVED else vocab


def _sample_cats(rng: np.random.Generator, tspec: S.TypeSpec, n: int) -> np.ndarray:
    """``[n, C]`` int32 categorical block — each column uniform over its legal index range."""
    cols = []
    for col_name, vocab in tspec.cat_cols:
        hi = _cat_high(tspec.name, col_name, vocab)
        cols.append(rng.integers(0, max(1, hi), size=n))
    return np.stack(cols, axis=-1).astype(np.int32) if cols else np.zeros((n, 0), np.int32)


def _sample_nums(rng: np.random.Generator, type_name: str, col_names: List[str], n: int) -> np.ndarray:
    """``[n, W]`` float32 numeric block. Each column's integer is sampled uniformly inside its measured
    ``spec.NUMERIC_RANGES`` (a flag column with no range is 0/1), then stored symlog (non-flag) / raw
    (flag) — the exact tokenizer mapping."""
    raw = RAW_NUM_COLS.get(type_name, set())
    ints = np.zeros((n, len(col_names)), dtype=np.int64)
    for j, c in enumerate(col_names):
        rng_spec = S.NUMERIC_RANGES.get(type_name, {}).get(c)
        lo, hi = (rng_spec.lo, rng_spec.hi) if rng_spec is not None else (0, 1)
        ints[:, j] = rng.integers(lo, hi + 1, size=n)
    out = ints.astype(np.float32)
    for j, c in enumerate(col_names):
        if c not in raw:
            out[:, j] = _symlog_arr(ints[:, j].astype(np.float64)).astype(np.float32)
    return out


def _left_pack_counts(rng: np.random.Generator, B: int, cap: int, size_sampler) -> np.ndarray:
    """Per-sample present count in ``[0, cap]`` via ``size_sampler(B)`` (clamped to cap)."""
    k = np.asarray(size_sampler(B))
    return np.clip(k, 0, cap).astype(np.int64)


# ==================================================================================================
# Per-expert fillers — write one category's designed arrays into a template batch, in place.
# ==================================================================================================

def _fill_scalars(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    gspec = S.TYPE_BY_NAME["global"]
    z["global_idx"][:, 0, :] = _sample_cats(rng, gspec, B)
    z["global_num"][:, 0, :] = _sample_nums(rng, "global", tokens.GLOBAL_NUM, B)
    # pending: [present(flag), minSelect(symlog), maxSelect(symlog), isUpgradeSelection(flag)].
    present = rng.integers(0, 2, size=B)
    mn = rng.integers(*_lohi("pending", "minSelect"), size=B)
    mx = rng.integers(*_lohi("pending", "maxSelect"), size=B)
    up = rng.integers(0, 2, size=B)
    pend = np.zeros((B, 4), np.float32)
    pend[:, 0] = present.astype(np.float32)
    pend[:, 1] = _symlog_arr(mn.astype(np.float64))
    pend[:, 2] = _symlog_arr(mx.astype(np.float64))
    pend[:, 3] = up.astype(np.float32)
    # A pending token that is absent (present flag 0) carries a zeroed block, matching tokenize().
    pend[present == 0] = 0.0
    z["pending"][:, 0, :] = pend


def _lohi(type_name: str, col: str) -> Tuple[int, int]:
    r = S.NUMERIC_RANGES[type_name][col]
    return r.lo, r.hi + 1


def _fill_potions(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Potion belt: random slot count 0..POTION_MAX_BELT; the belt is LEFT-PACKED and CANONICAL (v4) —
    the non-empty potions come first, sorted by catalog index (duplicates legal), then the index-0 empty
    slots, preserving belt SIZE. This mirrors :func:`tokens._canonical_from_state`, which canonicalizes
    slot position away because it is decision-irrelevant. The space is therefore the set of (potion
    MULTISET, belt size) pairs — exactly what the well-posed potion expert must cover; fully-empty belts
    and rare id combos are still drawn, only the impossible non-left-packed interleavings are removed. The
    belt caps at the game range (corpus max 5 + margin; a belt of >5 slots is not a valid configuration),
    well within the MAX_POTIONS padded dim."""
    cap = POTION_MAX_BELT
    n_potions = S.TYPE_BY_NAME["potion"].cat_cols[0][1]     # catalog size (index 0 = empty slot)
    counts = rng.integers(0, cap + 1, size=B)
    p_empty = rng.random(B)                                 # per-belt empty-slot probability (design)
    idx = z["potion_idx"]
    mask = z["potion_mask"]
    for b in range(B):
        k = int(counts[b])
        if k == 0:
            continue
        empt = rng.random(k) < p_empty[b]
        ids = rng.integers(1, n_potions, size=k)            # a real potion id
        ids[empt] = 0                                        # this slot is an empty belt slot
        n_empty = int(empt.sum())
        # Left-pack: non-empty ids sorted (canonical order), then the empty (id-0) slots.
        packed = np.concatenate([np.sort(ids[ids != 0]), np.zeros(n_empty, dtype=ids.dtype)])
        idx[b, :k, 0] = packed
        mask[b, :k] = True


RELIC_MAX_SET = 12   # game states hold 0..8 relics (measured data/corpus2 max 8) + margin; a long run can
                     # accumulate more (this corpus is act-0..2 homogeneous), so 12 covers the near future
                     # while staying well under the MAX_RELICS=40 padded cap. Sampling the full 40-slot cap
                     # made the synthetic task combinatorially harder than any real state.
RELIC_DUP_PROB = 0.05   # per-instance probability a drawn relic REPEATS an earlier one (duplicates are
                        # legal, rare — measured 3238/4.0M states carry one, max 2 copies). Small, with a
                        # natural rare high-count tail (>=2 repeats compound), so duplicates are LEARNED.


def _fill_relics(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Relics (v5 POSITIONAL): random count 0..RELIC_MAX_SET of relic ids at explicit slots 0..k-1, in
    generated (== wire) order. Ids drawn 1..N-1 (a present relic is a real relic). Duplicates are LEGAL
    and injected on purpose (:data:`RELIC_DUP_PROB`) so the expert learns them; order is preserved (never
    sorted) and the `slot` categorical == the row index (mirrors the tokenizer's positional stamp — the
    same treatment as orbs, so the permutation-invariant expert can see the semantic acquisition order)."""
    cap = RELIC_MAX_SET
    tspec = S.TYPE_BY_NAME["relic"]
    slot_col = [c for c, _ in tspec.cat_cols].index("slot")
    n_relics = tspec.cat_cols[0][1]
    counts = rng.integers(0, cap + 1, size=B)
    idx = z["relic_idx"]
    mask = z["relic_mask"]
    for b in range(B):
        k = int(counts[b])
        if k == 0:
            continue
        ids: List[int] = []
        for _ in range(k):
            if ids and rng.random() < RELIC_DUP_PROB:
                ids.append(int(rng.choice(ids)))            # duplicate an already-held relic (legal)
            else:
                ids.append(int(rng.integers(1, n_relics)))  # a fresh real relic id
        idx[b, :k, 0] = np.asarray(ids, dtype=np.int64)     # wire order preserved (NOT sorted)
        idx[b, :k, slot_col] = np.arange(k)                 # positional slot == list index
        mask[b, :k] = True


# Reachability-shaped orb space (owner-directed, 2026-07-18). Uniform sampling over the 32 hashed
# id buckets x independent [0..250] values trained the expert on a ~90-bit universe of mostly
# IMPOSSIBLE orbs (the game has 5 orb types with small type-conditional value ranges) - the flat
# convergence curve was the task design, not the model. Types below are the corpus-observed set
# with per-type (passive, evoke) ranges extended ~1.5x for margin; ORB_WILDCARD_PROB keeps a thin
# uniform tail over the full vocab/ranges as coverage insurance for unseen orb types.
ORB_TYPES = {
    # id string -> ((passive lo, hi), (evoke lo, hi)); ranges = observed corpus2 max * ~1.5 margin.
    "LIGHTNING_ORB": ((0, 15), (0, 23)),
    "FROST_ORB": ((0, 12), (0, 17)),
    "DARK_ORB": ((0, 20), (0, 102)),
    "PLASMA_ORB": ((1, 2), (2, 3)),
    "GLASS_ORB": ((0, 17), (0, 33)),
}
ORB_TYPE_BUCKETS = None   # resolved lazily: {hash bucket: (p_range, e_range)}
ORB_WILDCARD_PROB = 0.05
ORB_MAX_BELT = 12         # observed max 7 + margin (owner: ~a dozen slots possible, above is rare)


def _orb_buckets():
    global ORB_TYPE_BUCKETS
    if ORB_TYPE_BUCKETS is None:
        from .. import catalog as _cat
        ORB_TYPE_BUCKETS = {_cat.stable_hash(k, tokens.ORB_VOCAB): v for k, v in ORB_TYPES.items()}
    return ORB_TYPE_BUCKETS


def _fill_orbs(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Orbs: reachability-shaped — type from the real orb set (hashed to the same buckets the
    tokenizer uses), passive/evoke uniform within that TYPE's range (+margin); a thin
    ORB_WILDCARD_PROB tail samples any bucket/full ranges for coverage insurance. Positional slot
    column == belt index (well-posedness); empty belts included."""
    tspec = S.TYPE_BY_NAME["orb"]
    slot_col = [c for c, _ in tspec.cat_cols].index("slot")
    orb_col = [c for c, _ in tspec.cat_cols].index("orb")
    buckets = _orb_buckets()
    bucket_ids = list(buckets.keys())
    cap = min(ORB_MAX_BELT, tokens.MAX_ORBS)
    counts = rng.integers(0, cap + 1, size=B)
    idx = z["orb_idx"]
    num = z["orb_num"]
    mask = z["orb_mask"]
    p_i = tokens.ORB_NUM.index("passiveValue")
    e_i = tokens.ORB_NUM.index("evokeValue")
    for b in range(B):
        k = int(counts[b])
        if k == 0:
            continue
        cats = _sample_cats(rng, tspec, k)
        nums = _sample_nums(rng, "orb", tokens.ORB_NUM, k)
        for j in range(k):
            if rng.random() >= ORB_WILDCARD_PROB:
                bk = bucket_ids[int(rng.integers(len(bucket_ids)))]
                (plo, phi), (elo, ehi) = buckets[bk]
                cats[j, orb_col] = bk
                nums[j, p_i] = _symlog_arr(np.asarray([float(rng.integers(plo, phi + 1))]))[0]
                nums[j, e_i] = _symlog_arr(np.asarray([float(rng.integers(elo, ehi + 1))]))[0]
        cats[:, slot_col] = np.arange(k)                    # left-packed belt position == slot index
        idx[b, :k, :] = cats
        num[b, :k, :] = nums
        mask[b, :k] = True


_CREATURE_RAW = [c in RAW_NUM_COLS.get("creature", set()) for c in tokens.CREATURE_NUM]
_INTENT_RAW = [c in RAW_NUM_COLS.get("intent", set()) for c in tokens.INTENT_NUM]


def _hist_draw(rng: np.random.Generator, counts: Tuple[np.ndarray, np.ndarray], size: int) -> np.ndarray:
    """``size`` draws from a measured count histogram ``(values, probs)`` (creatures/powers/intents per
    state, powers per creature) — the vectorized twin of the old scalar ``_reach_hist``."""
    vals, probs = counts
    return rng.choice(vals, size=size, p=probs)


def _reach_num_block(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray,
                     is_raw: List[bool]) -> np.ndarray:
    """``[n, W]`` float block: one uniform integer per cell inside the gathered ``[lo, hi]`` reachability
    bounds, stored the way the tokenizer does (raw for flag columns, symlog otherwise). One ``rng.integers``
    over the whole block replaces the old per-row/per-column scalar draws."""
    vals = rng.integers(lo, hi + 1)                          # [n, W]
    out = vals.astype(np.float64)
    for j, raw in enumerate(is_raw):
        if not raw:
            out[:, j] = _symlog_arr(out[:, j])
    return out.astype(np.float32)


def _sample_creature_powers(rng: np.random.Generator, tbl: Dict[str, Any], pw_spec: S.TypeSpec,
                            pc: np.ndarray, bstate: np.ndarray, post_slot: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw each creature's DISTINCT powerIndex set + amounts. A power stacks its amount rather than
    duplicating, so two rows on one creature can never share a powerIndex — unreachable in the real game.
    Ids are drawn per creature by weighted sampling-WITHOUT-replacement (Gumbel-top-k) whose per-id weight
    is exactly the old per-row reachability marginal: the observed ``power_ids`` uniform mass plus a thin
    :data:`CREATURE_WILDCARD_PROB` full-vocab tail. So the wildcard/table behavior is preserved — only the
    duplicates are removed. Each row's amount is uniform in that powerIndex's reachability range (the full
    spec range for a wildcard id, monotone-safe because amounts can be negative). Returns flat creature-
    major ``(state, parent, powerIndex, amount)`` row arrays (``parent`` == the creature's canonical slot)."""
    Nc = pc.shape[0]
    maxk = int(pc.max()) if Nc else 0
    if maxk == 0:
        e = np.empty(0, np.int64)
        return e, e, e, e
    vocab = pw_spec.cat_cols[0][1]
    power_ids = tbl["power_ids"]
    w = CREATURE_WILDCARD_PROB
    weight = np.full(vocab, w / vocab, np.float64)            # wildcard tail: every id reachable
    weight[power_ids] += (1.0 - w) / len(power_ids)           # observed mass (== old per-row marginal)
    with np.errstate(divide="ignore"):
        logit = np.log(weight)                               # -inf for a zero-weight id (wildcard off)
    keys = logit[None, :] + rng.gumbel(size=(Nc, vocab))     # Gumbel-top-k == weighted WOR draw order
    if maxk < vocab:
        top = np.argpartition(-keys, maxk - 1, axis=1)[:, :maxk]
    else:
        top = np.broadcast_to(np.arange(vocab), (Nc, vocab)).copy()
    tk = np.take_along_axis(keys, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-tk, axis=1), axis=1)   # highest-weight id first, per creature
    keep = np.arange(maxk)[None, :] < pc[:, None]            # each creature keeps its first pc[f] ids
    pid = top[keep]                                          # flat, creature-major (distinct within group)
    amt = rng.integers(tbl["power_amt_lo_by_id"][pid], tbl["power_amt_hi_by_id"][pid] + 1)
    return np.repeat(bstate, pc), np.repeat(post_slot, pc), pid, amt


def _place_powers(pw_idx: np.ndarray, pw_num: np.ndarray, pw_mask: np.ndarray,
                  pstate: np.ndarray, pparent: np.ndarray, pid: np.ndarray, amt: np.ndarray,
                  B: int) -> None:
    """Scatter a flat block of power rows into the padded arrays in the tokenizer's CANONICAL flatten
    order: grouped by (state, parent creature slot) ascending, then within a creature by (powerIndex,
    amount) ascending — the exact order :func:`tokens.tokenize` emits (parent-slot flatten of powers each
    creature already sorted by ``(idx, amount)``). ``amt`` is the sampled INTEGER, so the sort is exact
    even for negative amounts (symlog is monotone, but sorting the integer avoids any float doubt). The
    per-state MAX_POWERS cap is applied in that slot order (== truncating ``powers[:MAX_POWERS]``)."""
    n = pstate.shape[0]
    if n == 0:
        return
    order = np.lexsort((amt, pid, pparent, pstate))          # primary state, then parent, idx, amount
    ps, pp, pi, am = pstate[order], pparent[order], pid[order], amt[order]
    rank = np.arange(n) - _exclusive_starts(np.bincount(ps, minlength=B), B)[ps]
    keep = rank < tokens.MAX_POWERS                          # cap in canonical slot order
    ps, pp, pi, am, rank = ps[keep], pp[keep], pi[keep], am[keep], rank[keep]
    pw_idx[ps, rank, 0] = pi
    pw_idx[ps, rank, 1] = pp
    pw_num[ps, rank, 0] = _symlog_arr(am.astype(np.float64)).astype(np.float32)
    pw_mask[ps, rank] = True


def _place_intents(rng: np.random.Generator, tbl: Dict[str, Any], in_spec: S.TypeSpec,
                   in_idx: np.ndarray, in_num: np.ndarray, in_mask: np.ndarray,
                   istate: np.ndarray, parent: np.ndarray, B: int) -> None:
    """Assign type + numerics for a flat block of intent rows (per-type numeric ranges gathered in bulk; a
    uniform wildcard tail), then scatter them in the tokenizer's CANONICAL flatten order: grouped by
    (state, parent creature slot) ascending, then within a creature by (type, damage, baseDamage, hits)
    ascending — the numerics compared as decoded INTEGERS, matching :func:`tokens._creature_canonical`.
    Parent ASSIGNMENT stays as passed (random creature slot); only the ROW ORDER is made canonical."""
    n = istate.shape[0]
    if n == 0:
        return
    intent_ids = tbl["intent_ids"]
    W = len(tokens.INTENT_NUM)
    wild = rng.random(n) < CREATURE_WILDCARD_PROB
    ty = np.empty(n, np.int64)
    numblk = np.zeros((n, W), np.float32)
    nw = int(wild.sum())
    if nw:
        ty[wild] = rng.integers(0, _cat_high("intent", "type", in_spec.cat_cols[0][1]), size=nw)
        numblk[wild] = _sample_nums(rng, "intent", tokens.INTENT_NUM, nw)
    tabm = ~wild
    nt = int(tabm.sum())
    if nt:
        ip = rng.integers(0, len(intent_ids), size=nt)
        ty[tabm] = intent_ids[ip]
        numblk[tabm] = _reach_num_block(rng, tbl["intent_num_lo"][ip], tbl["intent_num_hi"][ip], _INTENT_RAW)
    # Decode the sort-key numerics (damage/baseDamage/hits are symlog-stored) back to integers so the row
    # order matches the tokenizer's integer comparator exactly. INTENT_NUM = [hasDamage, damage,
    # baseDamage, hasHits, hits]; the intent key is (type, damage, baseDamage, hits).
    dec = numblk.astype(np.float64)
    ints = np.rint(np.where(_INTENT_RAW, dec, _symexp_arr(dec))).astype(np.int64)
    order = np.lexsort((ints[:, 4], ints[:, 2], ints[:, 1], ty, parent, istate))
    ist, par, tt, nb = istate[order], parent[order], ty[order], numblk[order]
    rank = np.arange(n) - _exclusive_starts(np.bincount(ist, minlength=B), B)[ist]
    in_idx[ist, rank, 0] = tt
    in_idx[ist, rank, 1] = par
    in_num[ist, rank, :] = nb
    in_mask[ist, rank] = True


# ==================================================================================================
# Creature-family fillers (owner ruling, 2026-07-18): the old single _fill_creatures (creature+power+intent
# in one expert) is split into three — creature-stats / creature-powers / creature-intents — one per token
# type, so each sub-task can be trained + measured on its own. The split preserves the e2c6e83 canonicality
# contract EXACTLY (it reuses the same _sample_creature_powers / _place_powers / _place_intents code paths
# and the same v4 creature lexsort): no ordering logic is re-derived here.
#
# CONSISTENCY across the split: powers/intents reference a PARENT creature slot, which must be a valid
# 0..c-1 index for the state's creature count c. The three fillers therefore share ONE per-state creature
# count (:func:`_draw_creature_counts`), drawn once by :func:`synth_batch` when >1 of them is requested
# together — so parents stay consistent with the creature rows creature-stats writes, and the combined draw
# order (stats -> powers -> intents, after the shared count) is byte-identical to the old _fill_creatures.
# A LONE creature-powers / creature-intents run is trainable STANDALONE: it draws its OWN virtual
# creature-count context from the same histogram, so its parent slots (and the MAX_POWERS/state cap) stay
# realistic even with no creature rows present.
# ==================================================================================================

def _draw_creature_counts(rng: np.random.Generator, tbl: Dict[str, Any], B: int) -> np.ndarray:
    """Per-state creature count: >=1 (so powers/intents always have a valid parent slot), capped to
    MAX_CREATURES, from the measured creatures-per-state histogram. Drawn ONCE and shared across the three
    creature-family fillers (see the section header)."""
    return np.clip(np.maximum(_hist_draw(rng, tbl["counts"]["creatures_per_state"], B), 1),
                   1, tokens.MAX_CREATURES).astype(np.int64)


def _fill_creature_stats(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int,
                         c_arr: Optional[np.ndarray] = None) -> None:
    """creature-stats expert (owns the `creature` token type) — REACHABILITY-SHAPED creature rows only (no
    powers/intents; those are their own experts now). Creature count from the shared context (or drawn here
    when run alone). Per creature: identity from the observed set, kind from that identity's observed kinds,
    numerics uniform in the identity's observed range (+margin, clamped); a :data:`CREATURE_WILDCARD_PROB`
    uniform tail. Rows are v4-lexsorted STATE-MAJOR — the tokenizer's canonical order (unchanged in v5).
    Creature ORDER is semantic (the pending v6 positional change), deliberately NOT modelled here.

    Fully VECTORIZED: the whole batch of creatures is generated as one flat state-major block and lexsorted
    in a SINGLE call, then scattered to slots — no per-row Python in the hot path."""
    tbl = _load_reachable()
    creatures = tbl["creatures"]
    creature_ids = tbl["creature_ids"]
    cr_spec = S.TYPE_BY_NAME["creature"]
    cr_idx, cr_num, cr_mask = z["creature_idx"], z["creature_num"], z["creature_mask"]
    if c_arr is None:
        c_arr = _draw_creature_counts(rng, tbl, B)
    Nc = int(c_arr.sum())
    bstate = np.repeat(np.arange(B, dtype=np.int64), c_arr)
    starts = _exclusive_starts(c_arr, B)

    cats = _sample_cats(rng, cr_spec, Nc)                    # [Nc, 2] uniform base (kind, identity)
    nums = _sample_nums(rng, "creature", tokens.CREATURE_NUM, Nc)   # [Nc, 5] symlog-stored base
    tab_pos = np.nonzero(rng.random(Nc) >= CREATURE_WILDCARD_PROB)[0]
    if tab_pos.shape[0]:
        pick = rng.integers(0, len(creature_ids), size=tab_pos.shape[0])
        cats[tab_pos, 1] = creature_ids[pick]               # CREATURE_IDX[1] = identity
        nums[tab_pos, :] = _reach_num_block(rng, tbl["creature_num_lo"][pick],
                                            tbl["creature_num_hi"][pick], _CREATURE_RAW)
        uniq, inv = np.unique(pick, return_inverse=True)     # kind is a per-identity variable-length list
        for ui, pidx in enumerate(uniq.tolist()):
            rows = tab_pos[inv == ui]
            cats[rows, 0] = rng.choice(creatures[int(creature_ids[pidx])]["kind"], size=rows.shape[0])

    # Canonical v4 creature order (kind, combatId, identity, currentHp, maxHp, block, active), state-major.
    # symlog is monotonic so sorting the stored floats matches the integer order. Key columns: kind=cats0,
    # combatId=nums4, identity=cats1, currentHp=nums0, maxHp=nums1, block=nums2, active=nums3.
    order = np.lexsort((nums[:, 3], nums[:, 2], nums[:, 1], nums[:, 0], cats[:, 1],
                        nums[:, 4], cats[:, 0], bstate))
    bs = bstate[order]
    dstslot = np.arange(Nc) - starts[bs]
    cr_idx[bs, dstslot] = cats[order]
    cr_num[bs, dstslot] = nums[order]
    cr_mask[bs, dstslot] = True


def _fill_creature_powers(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int,
                          c_arr: Optional[np.ndarray] = None) -> None:
    """creature-powers expert (owns the `power` token type). A per-creature power count from the
    powers-per-creature histogram (capped so the state total <= MAX_POWERS); powerIndex DISTINCT within a
    creature (a power STACKS its amount, never duplicates — an unreachable state; see
    :func:`_sample_creature_powers`); amount from that power's observed range (+margin — amounts may be
    negative, hence order-free -> sorted). A :data:`CREATURE_WILDCARD_PROB` uniform tail. Rows are placed
    in the tokenizer's CANONICAL (state, parent slot, powerIndex, amount) order via the shared
    :func:`_place_powers` path (the e2c6e83 contract, reused verbatim), with the per-state MAX_POWERS cap
    applied in slot order.

    PARENT slots: each flat creature is assigned its own within-state slot 0..c-1 (an identity assignment).
    Because creatures are exchangeable (the positional v6 change is deferred) and :func:`_place_powers`
    re-sorts by parent anyway, this is distributionally identical to the pre-split path (which pointed at
    the post-lexsort creature slot), while keeping powers trainable STANDALONE from any creature rows."""
    tbl = _load_reachable()
    pw_spec = S.TYPE_BY_NAME["power"]
    if c_arr is None:
        c_arr = _draw_creature_counts(rng, tbl, B)
    Nc = int(c_arr.sum())
    if Nc == 0:
        return
    bstate = np.repeat(np.arange(B, dtype=np.int64), c_arr)
    starts = _exclusive_starts(c_arr, B)
    parent_slot = np.arange(Nc, dtype=np.int64) - starts[bstate]   # each flat creature's own slot 0..c-1
    pc = _hist_draw(rng, tbl["counts"]["powers_per_creature"], Nc).astype(np.int64)
    p_state, p_parent, p_pid, p_amt = _sample_creature_powers(rng, tbl, pw_spec, pc, bstate, parent_slot)
    _place_powers(z["power_idx"], z["power_num"], z["power_mask"], p_state, p_parent, p_pid, p_amt, B)


def _fill_creature_intents(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int,
                           c_arr: Optional[np.ndarray] = None) -> None:
    """creature-intents expert (owns the `intent` token type). A per-state intent count from the
    intents-per-state histogram (capped MAX_INTENTS); type from the observed set; numerics per-type range;
    a :data:`CREATURE_WILDCARD_PROB` uniform tail. Each intent is assigned a UNIFORM parent creature slot
    0..c-1 (intents are order-free within a creature — game ruling — so only the ROW ORDER is canonical,
    not the assignment); rows are placed in the tokenizer's CANONICAL (state, parent, type, damage,
    baseDamage, hits) order via the shared :func:`_place_intents` path (the e2c6e83 contract, reused
    verbatim). Trainable STANDALONE via its own virtual creature-count context (``c_arr``)."""
    tbl = _load_reachable()
    in_spec = S.TYPE_BY_NAME["intent"]
    if c_arr is None:
        c_arr = _draw_creature_counts(rng, tbl, B)
    ni = np.minimum(_hist_draw(rng, tbl["counts"]["intents_per_state"], B),
                    tokens.MAX_INTENTS).astype(np.int64)
    Ni = int(ni.sum())
    if Ni:
        istate = np.repeat(np.arange(B, dtype=np.int64), ni)
        parent = (rng.random(Ni) * c_arr[istate]).astype(np.int64)   # uniform creature slot 0..c-1
        _place_intents(rng, tbl, in_spec, z["intent_idx"], z["intent_num"], z["intent_mask"],
                       istate, parent, B)


def _sample_from_hist(rng: np.random.Generator, hist: np.ndarray, size: int) -> np.ndarray:
    p = hist / hist.sum()
    return rng.choice(len(hist), size=size, p=p)


_CARD_RAW_COLS = RAW_NUM_COLS.get("card", set())
# (numeric-block index, column name, is-raw-flag) for every CARD_NUM content column — the dynamic numerics
# the reachability table conditions on. v6: CARD_NUM is exactly the 14 content numerics (no per-zone count
# columns), so this is the whole numeric block. (Name kept for the reachable/convergence twin references.)
_CARD_NONZONE_NUM = [(j, c, c in _CARD_RAW_COLS) for j, c in enumerate(tokens.CARD_NUM)]
# CARD_IDX categorical columns that are part of a card's CONTENT identity (the content-sort key) — i.e.
# everything except the v6 layout columns `zone` (the zone-major axis) and `slot` (the layout index).
_CARD_CONTENT_CAT_COLS = [i for i, c in enumerate(tokens.CARD_IDX) if c not in ("zone", "slot")]


def _card_content_order_ref(ci_b: np.ndarray, cn_b: np.ndarray, ckw_b: np.ndarray, k: int) -> List[int]:
    """Reference (per-row Python) row permutation that sorts the first ``k`` synthetic card rows by the
    SAME content key the tokenizer orders population rows with (:func:`tokens._card_content_key`). Kept
    verbatim as the ORACLE the vectorized :func:`_card_content_order` is proven identical to (see
    ``test_wm_synth.test_card_content_order_matches_reference``): the canonical row order is a wire-format
    contract with the tokenizer, so the fast path must reproduce it bit-for-bit. The key excludes the
    per-zone count columns (they are the count vector, not part of a row's content identity)."""
    keys = []
    for r in range(k):
        d = {name: int(ci_b[r, j]) for j, name in enumerate(tokens.CARD_IDX)}
        for j, name in enumerate(tokens.CARD_NUM):
            v = float(cn_b[r, j])
            d[name] = int(round(v)) if name in _CARD_RAW_COLS else int(round(tokens.symexp(v)))
        d["keywords"] = sorted(int(b) for b in np.nonzero(ckw_b[r])[0])
        keys.append(tokens._card_content_key(d))
    return sorted(range(k), key=lambda r: keys[r])


def _card_content_key_columns(ci: np.ndarray, cn: np.ndarray, ckw: np.ndarray) -> List[np.ndarray]:
    """Integer lexsort-key columns reproducing :func:`tokens._card_content_key` for a whole block of
    rows, MOST-significant first — the vectorized twin of the per-row tuple key.

    * The six CONTENT categoricals (cardIndex..afflict) pass through as-is. v6's `zone`/`slot` columns are
      EXCLUDED — the content key never carried zone (it is now the zone-major layout axis, handled
      separately) or slot (the layout index), exactly as :func:`tokens._card_content_key`.
    * Each dynamic numeric is decoded to the exact integer the tuple key carries: raw flags round; symlog
      columns invert (``round(symexp)``). symlog is monotonic so this is order-preserving, but decoding
      to the integer keeps ties/values byte-identical to the reference.
    * The keyword multiset (``sorted(present columns)``, compared as a tuple) becomes len(KEYWORDS)
      fixed-width columns holding the sorted-ascending set-column indices, RIGHT-padded with -1. A shorter
      sorted list is lexicographically smaller, so an absent slot (-1) must sort BEFORE any real column
      (0..6) — exactly the order Python gives ``tuple(a) < tuple(b)`` for the two sorted lists. (A single
      packed bitmask can NOT reproduce this: e.g. ``() < (0,) < (0,1) < (1,)`` is not a bitmask order.)
      v7: the columns are the 7 named ABSOLUTE keyword flags, not the old 32 hashed buckets."""
    kdim = len(tokens.KEYWORDS)
    cols: List[np.ndarray] = [ci[:, j].astype(np.int64) for j in _CARD_CONTENT_CAT_COLS]
    for j, _col, is_raw in _CARD_NONZONE_NUM:
        v = cn[:, j].astype(np.float64)
        dec = np.rint(v) if is_raw else np.rint(np.sign(v) * np.expm1(np.abs(v)))
        cols.append(dec.astype(np.int64))
    # Sorted-ascending keyword-column indices per row, padded with -1: set columns get their index, absent
    # columns get kdim (sorts to the tail), then the tail is rewritten to -1 (sorts to the front on
    # comparison, matching "shorter tuple is smaller").
    present = ckw > 0
    grid = np.where(present, np.arange(kdim, dtype=np.int64)[None, :], kdim)
    srt = np.sort(grid, axis=1)
    srt = np.where(srt == kdim, -1, srt)
    cols.extend(srt[:, c] for c in range(kdim))
    return cols


def _card_content_order(ci_b: np.ndarray, cn_b: np.ndarray, ckw_b: np.ndarray, k: int) -> List[int]:
    """Vectorized drop-in for :func:`_card_content_order_ref` — one ``np.lexsort`` over the integer key
    columns (:func:`_card_content_key_columns`) instead of building a Python tuple per row. ``np.lexsort``
    is stable and its last key is primary, so passing the columns most-significant-LAST (``[::-1]``)
    reproduces Python's tuple order, and equal-key rows keep their original relative order exactly as
    ``sorted()`` does."""
    if k <= 1:
        return list(range(k))
    cols = _card_content_key_columns(ci_b[:k], cn_b[:k], ckw_b[:k])
    return np.lexsort(cols[::-1]).tolist()


def _exclusive_starts(counts: np.ndarray, B: int) -> np.ndarray:
    """Per-state start offset into a flat (state-major) row array: ``[0, c0, c0+c1, …]``."""
    starts = np.zeros(B, np.int64)
    if B > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    return starts


def _fill_card_table_rows(rng: np.random.Generator, tbl: Dict[str, Any], cats: np.ndarray,
                          num: np.ndarray, kw: np.ndarray, tab_pos: np.ndarray) -> None:
    """Overwrite the table (non-wildcard) card rows in-place with reachability-shaped content. Numerics
    are drawn for the WHOLE block in one ``rng.integers`` over the identities' gathered bounds; the
    identity-conditioned categoricals + keyword pattern are drawn per DISTINCT identity (a handful of
    ``rng.choice`` calls each) rather than per row — the vectorization that replaces the old per-row
    Python table lookup."""
    n_tab = tab_pos.shape[0]
    card_ids = tbl["card_ids"]
    pick = rng.integers(0, len(card_ids), size=n_tab)        # index into card_ids (uniform == rng.choice)
    ar = np.arange(n_tab)
    cats[tab_pos, 0] = card_ids[pick]                        # CARD_IDX[0] = cardIndex
    # Numerics: gather each drawn identity's (lo, hi) and draw the block at once, symlog/raw-store per col.
    vals = rng.integers(tbl["card_num_lo"][pick], tbl["card_num_hi"][pick] + 1)   # [n_tab, n_nonzone]
    for j, (col_j, _col, is_raw) in enumerate(_CARD_NONZONE_NUM):
        cv = vals[:, j].astype(np.float64)
        num[tab_pos, col_j] = (cv if is_raw else _symlog_arr(cv)).astype(np.float32)
    # Uniform identity-conditioned categoricals: gather the padded option table, pick a random slot < count.
    for col_i, key in ((1, "type"), (2, "rarity"), (3, "targetType")):
        vmat = tbl["card_" + key + "_vals"][pick]                                 # [n_tab, L]
        pos = (rng.random(n_tab) * tbl["card_" + key + "_len"][pick]).astype(np.int64)
        cats[tab_pos, col_i] = vmat[ar, pos]
    # Weighted categoricals (enchant/afflict): inverse-CDF gather — first slot whose cumulative prob >= u.
    for col_i, key in ((4, "enchant"), (5, "afflict")):
        vmat = tbl["card_" + key + "_vals"][pick]
        pos = np.argmax(tbl["card_" + key + "_cdf"][pick] >= rng.random((n_tab, 1)), axis=1)
        cats[tab_pos, col_i] = vmat[ar, pos]
    # Keyword pattern: inverse-CDF pick by OBSERVED FREQUENCY (first pattern whose cumulative prob >= u) from
    # the identity's [pattern, len(KEYWORDS)] multi-hot table — NOT uniform over the deduped pattern set. The old
    # uniform draw turned a multi-pattern card into worst-case transmission load (the ~55% keyword residual);
    # weighting by frequency emits the canonical pattern at its true rate and keeps alternates rare.
    pat_idx = np.argmax(tbl["card_kw_cdf"][pick] >= rng.random((n_tab, 1)), axis=1)
    kw[tab_pos] = tbl["card_kw_mat"][pick, pat_idx]


def _fill_cards(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int,
                cards_max_rows: Optional[int] = None, bigdeck_boost: bool = False) -> None:
    """Card INSTANCE rows (v6) — REACHABILITY-SHAPED (see the reachable-table section header). One row per
    physical card copy: the instances-per-state COUNT is drawn from the measured
    ``counts["instances_per_state"]`` histogram; each instance's identity + static fields + dynamic numerics
    + keyword pattern come from that cardIndex's observed conditional table (values a real card actually
    takes, numerics widened ~1.5x and clamped), with a :data:`CARD_WILDCARD_PROB` per-row uniform tail as
    coverage insurance for unseen ids; and each instance's ``zone`` is drawn from the measured
    ``counts["card_zone"]`` marginal. Rows are then laid out ZONE-MAJOR (fixed ZONES order), within a zone
    CONTENT-SORTED into the tokenizer's canonical order, and stamped with ``slot`` == the layout index —
    the exact wire layout :func:`tokens.tokenize` produces (well-posed per-slot targets for the
    permutation-invariant card expert; the same treatment relics/orbs get).

    ``cards_max_rows`` (DIAGNOSTIC, ``--cards-max-rows N``, roadmap wm-t3-factored SINGLE-CARD probe):
    when set, the per-state instance count is overridden to ``min(N, drawn)`` and floored at 1 (never 0 —
    presence is still trained, and for ``N==1`` every state gets EXACTLY one card row). This isolates the
    per-row content machinery from all set effects (multi-row pooling / count reconstruction). ``None``
    (default) leaves the natural instances-per-state histogram untouched.

    ``bigdeck_boost`` (TRAIN-TIME only): with probability :data:`CARD_BIGDECK_BOOST` a state's instance count
    is REDRAWN from the renormalized ``>= CARD_BIGDECK_THRESH`` tail of the instances-per-state histogram, so
    large decks (which carry most real per-row error yet are rare) get extra training exposure. It is a
    deliberate departure from corpus frequency, applied ONLY on the training streams; ``coverage_val_sample``
    leaves it off so the coverage yardstick stays corpus-shaped. Ignored under the ``cards_max_rows`` probe.

    Fully VECTORIZED: all instances of all states are generated as one flat block (base uniform, then table
    rows overwritten, then zone drawn), and the whole batch is ordered in a SINGLE ``np.lexsort`` keyed by
    state -> zone -> content, then scattered into the padded arrays with ``slot`` stamped from the
    within-state position — no per-row Python in the hot path. The row ORDER matches the per-state reference
    (proven in ``test_wm_synth``); only the RNG draw order differs (distributional equivalence, per design)."""
    tbl = _load_reachable()
    cap = tokens.MAX_CARDS
    tspec = S.TYPE_BY_NAME["card"]
    zone_col = [c for c, _ in tspec.cat_cols].index("zone")
    slot_col = [c for c, _ in tspec.cat_cols].index("slot")
    ci, cn, ckw, cm = z["card_idx"], z["card_num"], z["card_kw"], z["card_mask"]

    inst_counts = np.clip(_hist_draw(rng, tbl["counts"]["instances_per_state"], B),
                          0, cap).astype(np.int64)
    if cards_max_rows is not None:
        # SINGLE-CARD probe cap: min(N, drawn), floored at 1 so presence is always trained (never 0 rows)
        # and N==1 yields exactly one row per state.
        inst_counts = np.clip(np.minimum(inst_counts, int(cards_max_rows)), 1, cap)
    elif bigdeck_boost:
        # Large-deck oversampling (train-time only): redraw a CARD_BIGDECK_BOOST fraction of states from the
        # renormalized >= CARD_BIGDECK_THRESH tail of the histogram (extra exposure of the error-heavy tail).
        ivals, iprobs = tbl["counts"]["instances_per_state"]
        tail = ivals >= CARD_BIGDECK_THRESH
        if tail.any():
            tvals, tp = ivals[tail], iprobs[tail] / iprobs[tail].sum()
            boost = rng.random(B) < CARD_BIGDECK_BOOST
            nb = int(boost.sum())
            if nb:
                inst_counts[boost] = np.clip(rng.choice(tvals, size=nb, p=tp), 0, cap).astype(np.int64)
    N = int(inst_counts.sum())
    if N == 0:
        return
    bstate = np.repeat(np.arange(B, dtype=np.int64), inst_counts)
    starts = _exclusive_starts(inst_counts, B)

    # Base content for every flat instance: uniform cardIndex/type/rarity/targetType (the wildcard tail keeps
    # this) + per-spec numerics. Keywords start EMPTY (no random bits) and table rows get their frequency-
    # weighted pattern below. cats' enchant/afflict start uniform but are pinned to 0 on wildcard rows.
    cats = _sample_cats(rng, tspec, N)                                          # [N, 8] (zone/slot rewritten)
    num = _sample_nums(rng, "card", tokens.CARD_NUM, N)                         # [N, 14] content numerics
    kw = np.zeros((N, len(tokens.KEYWORDS)), np.float32)
    tab_flags = rng.random(N) >= CARD_WILDCARD_PROB
    tab_pos = np.nonzero(tab_flags)[0]                                           # table (non-wildcard) rows
    if tab_pos.shape[0]:
        _fill_card_table_rows(rng, tbl, cats, num, kw, tab_pos)
    # STRUCTURED wildcard tail: uniform cardIndex over the full vocab (kept from _sample_cats) is id-level
    # insurance for unseen ids, with enchant/afflict pinned to 0 (their real marginal mode) — no
    # incompressible enchant/afflict bit noise. v7: a wildcard row's ABSOLUTE keywords are that cardIndex's
    # PRINTED keywords (the 'no runtime grants' absolute state) — gathered from the catalog; a cardIndex
    # outside the catalog (index 0 / unknown) maps to the all-zero row. NOT random bits. Numerics keep the
    # per-spec draw.
    enchant_col = [c for c, _ in tspec.cat_cols].index("enchant")
    afflict_col = [c for c, _ in tspec.cat_cols].index("afflict")
    wild_pos = np.nonzero(~tab_flags)[0]
    if wild_pos.shape[0]:
        cats[np.ix_(wild_pos, [enchant_col, afflict_col])] = 0
        kw[wild_pos] = tokens.printed_keyword_flags_by_index()[cats[wild_pos, 0]]
    # Zone per instance from the measured marginal (all rows — table AND wildcard), overwriting the base.
    cats[:, zone_col] = _hist_draw(rng, tbl["counts"]["card_zone"], N)

    # Order every state's instances ZONE-MAJOR then within-zone by content, in ONE lexsort (state primary,
    # then zone, then the content key most-significant-first). lexsort is stable, so identical copies keep
    # their generation order — a harmless tie. Then scatter to the within-state slot and stamp slot==index.
    order = np.lexsort(_card_content_key_columns(cats, num, kw)[::-1]
                       + [cats[:, zone_col].astype(np.int64), bstate])
    bs = bstate[order]
    dstslot = np.arange(N) - starts[bs]
    ci[bs, dstslot] = cats[order]
    ci[bs, dstslot, slot_col] = dstslot                     # positional slot == the layout index
    cn[bs, dstslot] = num[order]
    ckw[bs, dstslot] = kw[order]
    cm[bs, dstslot] = True


# The three creature-family experts share one per-state creature-count context (see the section header);
# synth_batch draws it once when >1 is requested together. Listed in canonical order (stats, powers,
# intents) so the combined draw sequence matches the old single _fill_creatures.
_CREATURE_EXPERTS = ("creature-stats", "creature-powers", "creature-intents")

_FILLERS = {
    "scalars": _fill_scalars, "potions": _fill_potions, "relics": _fill_relics,
    "orbs": _fill_orbs, "cards": _fill_cards,
    "creature-stats": _fill_creature_stats, "creature-powers": _fill_creature_powers,
    "creature-intents": _fill_creature_intents,
}


# ==================================================================================================
# Public API: batch generators + streams.
# ==================================================================================================

def synth_batch(experts: Iterable[str], batch_size: int, rng: np.random.Generator,
                cards_max_rows: Optional[int] = None,
                cards_bigdeck_boost: bool = False) -> Dict[str, np.ndarray]:
    """A full model-input batch (all :data:`model.BATCH_KEYS`, stacked ``[B, ...]``) with each named
    expert's category sampled uniformly-with-design and every other category left empty. ``experts`` are
    the trained expert names (:data:`experts.EXPERT_ORDER` keys).

    ``cards_max_rows`` (DIAGNOSTIC, ``--cards-max-rows N``): caps the cards generator's instances-per-state
    to ``min(N, drawn)`` floored at 1 (see :func:`_fill_cards`); ignored by every non-card expert.

    ``cards_bigdeck_boost``: enable the large-deck oversampling (:data:`CARD_BIGDECK_BOOST`) on the cards
    generator — set by the TRAINING streams, left OFF (default) by :func:`coverage_val_sample` so the
    coverage yardstick stays corpus-shaped. Ignored by every non-card expert."""
    z = _zeros_batch(batch_size)
    requested = list(experts)
    # Creature-family experts share ONE per-state creature-count context when >1 is requested together, so
    # the parent slots powers/intents reference stay consistent with the creature rows (and the combined
    # output matches the pre-split _fill_creatures distribution). A lone creature expert draws its own.
    creature_reqs = [e for e in _CREATURE_EXPERTS if e in requested]
    done: set = set()
    if len(creature_reqs) > 1:
        shared_c = _draw_creature_counts(rng, _load_reachable(), batch_size)
        for e in creature_reqs:                              # canonical order (stats, powers, intents)
            _FILLERS[e](rng, z, batch_size, shared_c)
            done.add(e)
    for e in requested:
        if e in done:
            continue
        filler = _FILLERS.get(e)
        if filler is None:
            raise KeyError(f"synth: unknown expert {e!r}; known {sorted(_FILLERS)}")
        if e == "cards":
            filler(rng, z, batch_size, cards_max_rows=cards_max_rows,
                   bigdeck_boost=cards_bigdeck_boost)
        else:
            filler(rng, z, batch_size)
    return z


def synth_batches(experts: List[str], batch_size: int, rng: np.random.Generator,
                  cards_max_rows: Optional[int] = None
                  ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite ``(stacked_numpy_batch, acts)`` stream of pure-synthetic batches (acts tagged ``"synth"``
    so the report's by-act group-by keeps them separable). ``cards_max_rows`` threads the SINGLE-CARD probe
    cap into every generated cards batch (see :func:`synth_batch`). This is a TRAINING stream, so the cards
    generator's large-deck oversampling (:data:`CARD_BIGDECK_BOOST`) is ON."""
    while True:
        yield (synth_batch(experts, batch_size, rng, cards_max_rows, cards_bigdeck_boost=True),
               ["synth"] * batch_size)


class _PrefetchError:
    """Envelope carrying a worker-thread exception across the queue so :func:`prefetch_batches` can
    re-raise it on the consumer side (a bare exception object could be confused with a real batch item)."""
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def prefetch_batches(inner: Iterator[Tuple[Dict[str, np.ndarray], List[Any]]], depth: int = 2
                     ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Overlap synthetic batch GENERATION with GPU compute: a daemon worker thread pulls ``(batch, acts)``
    tuples from ``inner`` into a bounded ``queue.Queue(depth)`` and the consumer yields them, so the next
    batch is being built on the CPU while the current one trains on the GPU (generation is otherwise serial
    with — and starves — the train step; measured ~19x on the cards generator).

    WHY a dedicated wrapper (vs the fire-and-forget :func:`wm.data.prefetch`): this one PROPAGATES a
    worker exception to the consumer — it is re-raised on the next ``get`` — so a generator bug fails the
    run loudly instead of silently ending the stream and starving the loop. The worker is a daemon (never
    blocks interpreter exit) and a sentinel on normal end is enough shutdown protocol.

    RNG note: ``inner`` keeps generating on this single worker thread using the ``np.random.Generator`` it
    already owns. That Generator must not be shared with any other concurrent consumer — the caller
    (``train_encdec``) builds a dedicated ``default_rng`` per synthetic stream, so it is not."""
    q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
    sentinel = object()

    def worker() -> None:
        try:
            for item in inner:
                q.put(item)
        except BaseException as ex:                          # propagate to the consumer rather than die silently
            q.put(_PrefetchError(ex))
        else:
            q.put(sentinel)

    threading.Thread(target=worker, name="synth-prefetch", daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, _PrefetchError):
            raise item.exc
        yield item


def mixed_batches(cache_dir: str, split: str, experts: List[str], batch_size: int, frac_synth: float,
                  rng: random.Random, cards_max_rows: Optional[int] = None
                  ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite ``(stacked_numpy_batch, acts)`` stream where a fraction ``frac_synth`` of each batch is
    synthetic and the remainder is real (read from the pre-tokenized cache via
    :func:`data.cache_batches_cpu`). The two halves are concatenated and shuffled so a batch has no
    real/synth ordering structure. ``cards_max_rows`` applies the SINGLE-CARD probe cap to the SYNTHETIC
    half only (the real half keeps its natural rows — see :func:`synth_batch`). The synthetic half is a
    TRAINING stream, so the cards generator's large-deck oversampling (:data:`CARD_BIGDECK_BOOST`) is ON."""
    from . import data as D
    n_synth = int(round(frac_synth * batch_size))
    n_real = batch_size - n_synth
    npr = np.random.default_rng(rng.getrandbits(64))
    real_stream = D.cache_batches_cpu(cache_dir, split, n_real, rng) if n_real > 0 else None
    while True:
        parts: List[Dict[str, np.ndarray]] = []
        acts: List[Any] = []
        if n_synth > 0:
            parts.append(synth_batch(experts, n_synth, npr, cards_max_rows, cards_bigdeck_boost=True))
            acts += ["synth"] * n_synth
        if real_stream is not None:
            r_arr, r_acts = next(real_stream)
            parts.append(r_arr)
            acts += list(r_acts)
        merged = {k: np.concatenate([p[k] for p in parts]) for k in M.BATCH_KEYS}
        order = npr.permutation(len(acts))
        merged = {k: merged[k][order] for k in M.BATCH_KEYS}
        acts = [acts[i] for i in order]
        yield merged, acts


def coverage_val_sample(experts: List[str], n: int, seed: int,
                        cards_max_rows: Optional[int] = None
                        ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """A FIXED, seeded synthetic coverage-val sample: ``n`` uniformly-with-design configurations for the
    named experts. Deterministic in ``seed`` so the coverage yardstick is identical across runs (the same
    role the real fixed val plays, on the synthetic space).

    ``cards_max_rows`` (DIAGNOSTIC, ``--cards-max-rows N``): honors the SAME SINGLE-CARD probe cap the
    training stream uses, so train and coverage-val measure the same restricted space (see
    :func:`synth_batch`).

    NOTE: the cards large-deck oversampling (:data:`CARD_BIGDECK_BOOST`) is deliberately LEFT OFF here — the
    coverage yardstick must stay corpus-shaped (unboosted), matching the real-val distribution, while the
    training stream boosts large decks above corpus frequency."""
    rng = np.random.default_rng(seed)
    return synth_batch(experts, n, rng, cards_max_rows), ["synth"] * n


# Fixed seed for the coverage-val sample — independent of the training seed so every run's coverage
# yardstick is the same 2000 configs.
COVERAGE_VAL_SEED = 0xC0FFEE
