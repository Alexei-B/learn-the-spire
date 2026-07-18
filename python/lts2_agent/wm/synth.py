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

import random
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


# ==================================================================================================
# Measured game-like marginals for the CARDS expert (coverage insurance — cards keep game-shaped
# population structure; only ids/dynamic-numerics are uniform). Measured 2026-07-17 over one train shard
# of data/corpus_tok_v3 (4096 states, 43,328 present population rows) — the same scan wm.ranges uses.
# Everything else (potions/relics/orbs/creatures) is uniform-with-design and needs no game marginal.
# ==================================================================================================

# Rows-per-state histogram, k = 0..23 (observed max 23; padded dim MAX_CARDS = 64). Sampled as the card
# population size so a synthetic state has a game-plausible number of distinct-content rows.
_CARD_ROWS_HIST = np.array(
    [33, 0, 1, 10, 100, 247, 389, 549, 412, 253, 104, 78, 186, 372, 411, 476, 236, 118, 64, 32, 20, 2,
     2, 1], dtype=np.float64)
# Number of distinct zones a single row occupies: {1: 40564, 2: 2607, 3: 155, 4: 2} (93.6% single-zone).
_CARD_NZONES_HIST = np.array([0, 40564, 2607, 155, 2], dtype=np.float64)   # index = n_zones (0 unused)
# Which zones a row tends to occupy (hand/draw/discard/exhaust/offered present-fraction over rows).
_CARD_ZONE_WEIGHTS = np.array([0.310, 0.349, 0.276, 0.117, 0.015], dtype=np.float64)
# Per-occupied-zone instance count (small int); index = count. From total-instances/row {1:.815 …}, most
# rows hold a single instance. Truncated + renormalized to the zone's measured range at sample time.
_CARD_ZONE_COUNT_HIST = np.array([0, 0.815, 0.100, 0.044, 0.024, 0.007, 0.010], dtype=np.float64)


# ==================================================================================================
# Empty full batch template (all BATCH_KEYS at their padded shapes, trivial empty encoding).
# ==================================================================================================

def _zeros_batch(B: int) -> Dict[str, np.ndarray]:
    z: Dict[str, np.ndarray] = {
        "global_idx": np.zeros((B, 1, len(tokens.GLOBAL_IDX)), np.int32),
        "global_num": np.zeros((B, 1, len(tokens.GLOBAL_NUM)), np.float32),
        "pending": np.zeros((B, 1, len(tokens.PENDING_NUM)), np.float32),
        "card_kw": np.zeros((B, tokens.MAX_CARDS, tokens.KW_BUCKETS), np.float32),
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


def _sample_nums(rng: np.random.Generator, type_name: str, col_names: List[str], n: int,
                 zone_cols: Optional[List[str]] = None) -> np.ndarray:
    """``[n, W]`` float32 numeric block. Each column's integer is sampled uniformly inside its measured
    ``spec.NUMERIC_RANGES`` (a flag column with no range is 0/1), then stored symlog (non-flag) / raw
    (flag) — the exact tokenizer mapping. ``zone_cols`` are skipped here (the CARDS generator fills the
    per-zone count vector with its own game-like small-int distribution)."""
    raw = RAW_NUM_COLS.get(type_name, set())
    zone_cols = zone_cols or []
    ints = np.zeros((n, len(col_names)), dtype=np.int64)
    for j, c in enumerate(col_names):
        if c in zone_cols:
            continue
        rng_spec = S.NUMERIC_RANGES.get(type_name, {}).get(c)
        lo, hi = (rng_spec.lo, rng_spec.hi) if rng_spec is not None else (0, 1)
        ints[:, j] = rng.integers(lo, hi + 1, size=n)
    out = ints.astype(np.float32)
    for j, c in enumerate(col_names):
        if c not in raw and c not in zone_cols:
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


def _fill_orbs(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Orbs: random slot count 0..MAX_ORBS; orb id uniform over the hashed vocab; passive/evoke values
    uniform inside their ranges (symlog-stored). Empty belts included. The `slot` categorical (v4) is the
    orb's belt POSITION — with left-packed presence a present orb at slot i has position i, so the slot
    column is ``arange(k)`` (mirrors the tokenizer's running index). Position is now a learnable field the
    encoder sees, so the semantic evoke order is representable (the well-posedness fix)."""
    tspec = S.TYPE_BY_NAME["orb"]
    slot_col = [c for c, _ in tspec.cat_cols].index("slot")
    cap = tokens.MAX_ORBS
    counts = rng.integers(0, cap + 1, size=B)
    idx = z["orb_idx"]
    num = z["orb_num"]
    mask = z["orb_mask"]
    for b in range(B):
        k = int(counts[b])
        if k == 0:
            continue
        cats = _sample_cats(rng, tspec, k)
        cats[:, slot_col] = np.arange(k)                    # left-packed belt position == slot index
        idx[b, :k, :] = cats
        num[b, :k, :] = _sample_nums(rng, "orb", tokens.ORB_NUM, k)
        mask[b, :k] = True


def _fill_creatures(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Creatures (folding their powers + intents). Random creature count 1..MAX_CREATURES; kind/identity
    uniform; numerics uniform in ranges. Powers/intents get uniform ids + in-range amounts and a VALID
    parent ref (a present creature slot). Counts respect the per-type caps."""
    cr_spec = S.TYPE_BY_NAME["creature"]
    pw_spec = S.TYPE_BY_NAME["power"]
    in_spec = S.TYPE_BY_NAME["intent"]
    cr_idx, cr_num, cr_mask = z["creature_idx"], z["creature_num"], z["creature_mask"]
    pw_idx, pw_num, pw_mask = z["power_idx"], z["power_num"], z["power_mask"]
    in_idx, in_num, in_mask = z["intent_idx"], z["intent_num"], z["intent_mask"]
    # At least one creature, so powers/intents always have a valid parent to reference.
    cr_counts = rng.integers(1, tokens.MAX_CREATURES + 1, size=B)
    for b in range(B):
        c = int(cr_counts[b])
        cats = _sample_cats(rng, cr_spec, c)                        # [c, 2] (kind, identity)
        nums = _sample_nums(rng, "creature", tokens.CREATURE_NUM, c)  # [c, 5], symlog-stored
        # Canonicalize creature order to match the tokenizer (v4): sort by (kind, combatId, identity,
        # currentHp, maxHp, block, active). symlog is monotonic so sorting the stored floats matches the
        # integer order. Powers/intents are generated AFTER placing, so their parent refs index the
        # sorted positions automatically (no remap). Key columns: kind=cats0, combatId=nums4,
        # identity=cats1, currentHp=nums0, maxHp=nums1, block=nums2, active=nums3.
        order = np.lexsort((nums[:, 3], nums[:, 2], nums[:, 1], nums[:, 0], cats[:, 1],
                            nums[:, 4], cats[:, 0]))
        cr_idx[b, :c, :] = cats[order]
        cr_num[b, :c, :] = nums[order]
        cr_mask[b, :c] = True
        # Powers.
        np_ = int(rng.integers(0, tokens.MAX_POWERS + 1))
        if np_:
            ids = rng.integers(0, pw_spec.cat_cols[0][1], size=np_)      # powerIndex over catalog
            parents = rng.integers(0, c, size=np_)                      # valid present-creature parent
            pw_idx[b, :np_, 0] = ids
            pw_idx[b, :np_, 1] = parents
            pw_num[b, :np_, :] = _sample_nums(rng, "power", tokens.POWER_NUM, np_)
            pw_mask[b, :np_] = True
        # Intents.
        ni = int(rng.integers(0, tokens.MAX_INTENTS + 1))
        if ni:
            types = rng.integers(0, _cat_high("intent", "type", in_spec.cat_cols[0][1]), size=ni)
            parents = rng.integers(0, c, size=ni)
            in_idx[b, :ni, 0] = types
            in_idx[b, :ni, 1] = parents
            in_num[b, :ni, :] = _sample_nums(rng, "intent", tokens.INTENT_NUM, ni)
            in_mask[b, :ni] = True


def _sample_from_hist(rng: np.random.Generator, hist: np.ndarray, size: int) -> np.ndarray:
    p = hist / hist.sum()
    return rng.choice(len(hist), size=size, p=p)


_CARD_RAW_COLS = RAW_NUM_COLS.get("card", set())


def _card_content_order(ci_b: np.ndarray, cn_b: np.ndarray, ckw_b: np.ndarray, k: int) -> List[int]:
    """Row permutation that sorts the first ``k`` synthetic card rows by the SAME content key the
    tokenizer orders population rows with (:func:`tokens._card_content_key`) — so synthetic card targets
    obey the identical canonical order as real tokenized states (generator-canonicality guard: synth
    bypasses tokenize, so it must reproduce the invariant itself). The key excludes the per-zone count
    columns (they are the count vector, not part of a row's content identity)."""
    keys = []
    for r in range(k):
        d = {name: int(ci_b[r, j]) for j, name in enumerate(tokens.CARD_IDX)}
        for j, name in enumerate(tokens.CARD_NUM):
            if name in tokens.ZONE_COUNT_FIELDS:
                continue
            v = float(cn_b[r, j])
            d[name] = int(round(v)) if name in _CARD_RAW_COLS else int(round(tokens.symexp(v)))
        d["keywords"] = sorted(int(b) for b in np.nonzero(ckw_b[r])[0])
        keys.append(tokens._card_content_key(d))
    return sorted(range(k), key=lambda r: keys[r])


def _fill_cards(rng: np.random.Generator, z: Dict[str, np.ndarray], B: int) -> None:
    """Card population rows (coverage insurance — game-shaped structure, uniform content). Row count from
    the measured rows/state marginal; each row's categoricals + dynamic numerics uniform in ranges;
    keywords random-sparse; the per-zone count vector sampled from the measured game-like small-int
    distribution (n_zones, which zones, per-zone count) with sum >= 1 (a row exists because it has an
    instance somewhere). Rows are finally CONTENT-SORTED into the tokenizer's canonical order (well-posed
    targets — a permutation-invariant set encoder needs a content-determined slot assignment)."""
    cap = tokens.MAX_CARDS
    tspec = S.TYPE_BY_NAME["card"]
    zone_cols = list(tokens.ZONE_COUNT_FIELDS)
    zone_maxes = {"count_hand": 20, "count_draw": 40, "count_discard": 40, "count_exhaust": 40,
                  "count_offered": 30}
    ci, cn, ckw, cm = z["card_idx"], z["card_num"], z["card_kw"], z["card_mask"]
    row_counts = np.clip(_sample_from_hist(rng, _CARD_ROWS_HIST, B), 0, cap)
    zone_count_idx = [tokens.CARD_NUM.index(zc) for zc in zone_cols]
    for b in range(B):
        k = int(row_counts[b])
        if k == 0:
            continue
        ci[b, :k, :] = _sample_cats(rng, tspec, k)
        cn[b, :k, :] = _sample_nums(rng, "card", tokens.CARD_NUM, k, zone_cols=zone_cols)
        # Per-row keyword multi-hot: sparse (each of KW_BUCKETS on with small prob).
        ckw[b, :k, :] = (rng.random((k, tokens.KW_BUCKETS)) < 0.05).astype(np.float32)
        # Per-zone count vector (game-like): pick n_zones, which zones, then a small count each.
        for r in range(k):
            nz = int(_sample_from_hist(rng, _CARD_NZONES_HIST, 1)[0])
            nz = max(1, min(nz, len(zone_cols)))
            chosen = rng.choice(len(zone_cols), size=nz, replace=False,
                                p=_CARD_ZONE_WEIGHTS / _CARD_ZONE_WEIGHTS.sum())
            for zi in chosen:
                zc = zone_cols[zi]
                cnt = int(_sample_from_hist(rng, _CARD_ZONE_COUNT_HIST, 1)[0])
                cnt = max(1, min(cnt, zone_maxes[zc]))
                cn[b, r, zone_count_idx[zi]] = _symlog_arr(np.float64(cnt))
        # Canonicalize row order to the tokenizer's content sort (well-posedness — see helper docstring).
        order = _card_content_order(ci[b], cn[b], ckw[b], k)
        ci[b, :k] = ci[b, order]
        cn[b, :k] = cn[b, order]
        ckw[b, :k] = ckw[b, order]
        cm[b, :k] = True


_FILLERS = {
    "scalars": _fill_scalars, "potions": _fill_potions, "relics": _fill_relics,
    "orbs": _fill_orbs, "creatures": _fill_creatures, "cards": _fill_cards,
}


# ==================================================================================================
# Public API: batch generators + streams.
# ==================================================================================================

def synth_batch(experts: Iterable[str], batch_size: int, rng: np.random.Generator
                ) -> Dict[str, np.ndarray]:
    """A full model-input batch (all :data:`model.BATCH_KEYS`, stacked ``[B, ...]``) with each named
    expert's category sampled uniformly-with-design and every other category left empty. ``experts`` are
    the trained expert names (:data:`experts.EXPERT_ORDER` keys)."""
    z = _zeros_batch(batch_size)
    for e in experts:
        filler = _FILLERS.get(e)
        if filler is None:
            raise KeyError(f"synth: unknown expert {e!r}; known {sorted(_FILLERS)}")
        filler(rng, z, batch_size)
    return z


def synth_batches(experts: List[str], batch_size: int, rng: np.random.Generator
                  ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite ``(stacked_numpy_batch, acts)`` stream of pure-synthetic batches (acts tagged ``"synth"``
    so the report's by-act group-by keeps them separable)."""
    while True:
        yield synth_batch(experts, batch_size, rng), ["synth"] * batch_size


def mixed_batches(cache_dir: str, split: str, experts: List[str], batch_size: int, frac_synth: float,
                  rng: random.Random) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite ``(stacked_numpy_batch, acts)`` stream where a fraction ``frac_synth`` of each batch is
    synthetic and the remainder is real (read from the pre-tokenized cache via
    :func:`data.cache_batches_cpu`). The two halves are concatenated and shuffled so a batch has no
    real/synth ordering structure."""
    from . import data as D
    n_synth = int(round(frac_synth * batch_size))
    n_real = batch_size - n_synth
    npr = np.random.default_rng(rng.getrandbits(64))
    real_stream = D.cache_batches_cpu(cache_dir, split, n_real, rng) if n_real > 0 else None
    while True:
        parts: List[Dict[str, np.ndarray]] = []
        acts: List[Any] = []
        if n_synth > 0:
            parts.append(synth_batch(experts, n_synth, npr))
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


def coverage_val_sample(experts: List[str], n: int, seed: int
                        ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """A FIXED, seeded synthetic coverage-val sample: ``n`` uniformly-with-design configurations for the
    named experts. Deterministic in ``seed`` so the coverage yardstick is identical across runs (the same
    role the real fixed val plays, on the synthetic space)."""
    rng = np.random.default_rng(seed)
    return synth_batch(experts, n, rng), ["synth"] * n


# Fixed seed for the coverage-val sample — independent of the training seed so every run's coverage
# yardstick is the same 2000 configs.
COVERAGE_VAL_SEED = 0xC0FFEE
