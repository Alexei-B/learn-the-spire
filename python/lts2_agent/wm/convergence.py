"""Convergence sanity-check: is each expert learning as fast as its task entropy predicts?

The owner's heuristic: steps-to-converge should follow ``f(H) = a * H^b`` where ``H`` is the
entropy (bits/state) of the synthetic space the expert compresses. Experts sitting far ABOVE the
fitted curve are flagged LAGGING — historically that has meant a bug (ill-posed ordering, generator
canonicality, loss miscalibration), not a hard task, so the flag is a tripwire, not a verdict.

Entropies are ANALYTIC APPROXIMATIONS computed from the same constants the synth generators use
(caps, vocab sizes, NUMERIC_RANGES). They ignore small multiset/duplicate reductions — documented
per formula — which is fine for a scaling heuristic (errors of a bit or two don't move a power-law
fit materially).

Milestones are read from run metrics (coverage-val by default — the training distribution — so the
comparison is apples-to-apples across experts; real-val milestones optional). The fit uses experts
that crossed a milestone; predictions + verdicts are emitted for everyone, including uncrossed
experts (predicted crossing step vs. steps trained so far).

Caveat recorded from the first use: the canary probes ran cosine-to-50k halted early, so late-run
milestone timings are LR-confounded. For comparable milestones, run probes with the --steps big +
--halt-step trick (near-flat LR over the probe window).

Usage::

    python -m lts2_agent.wm.convergence --runs wp5-relics-synth wp5-potions-synth wp5-orbs-synth
    python -m lts2_agent.wm.convergence --runs ... --metric exact --thresholds 0.25,0.5,0.9
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import spec as S
from . import synth as SY


# --------------------------------------------------------------------------------------------------
# Analytic task entropies (bits/state) from generator constants.
# --------------------------------------------------------------------------------------------------

def _range_bits(type_name: str, col: str) -> float:
    r = S.NUMERIC_RANGES.get(type_name, {}).get(col)
    if r is None:
        return 0.0
    n = int((r.hi - r.lo) / r.resolution) + 1
    return math.log2(max(2, n))


def entropy_potions() -> float:
    """H(belt size 0..cap) + E[k] * per-slot H(empty-vs-id marginal). Ignores the multiset
    reduction (overestimates by ~1-2 bits)."""
    cap = SY.POTION_MAX_BELT
    n_ids = S.TYPE_BY_NAME["potion"].cat_cols[0][1] - 1
    e_k = cap / 2.0
    h_slot = 1.0 + 0.5 * math.log2(n_ids)      # p(empty) marginal = 0.5 under p~U(0,1)
    return math.log2(cap + 1) + e_k * h_slot


def entropy_relics() -> float:
    """H(set size) + E[k] * log2(#ids). Positional rows: slot == index carries no extra entropy;
    duplicate allowance adds a negligible fraction of a bit."""
    cap = getattr(SY, "RELIC_MAX_SET", 12)
    n_ids = S.TYPE_BY_NAME["relic"].cat_cols[0][1] - 1
    return math.log2(cap + 1) + (cap / 2.0) * math.log2(n_ids)


def entropy_orbs() -> float:
    """Reachability-shaped space: H(count) + E[k] * (log2(#real types) + mean per-type value bits
    + wildcard tail). Matches the ORB_TYPES generator (2026-07-18)."""
    cap = getattr(SY, "ORB_MAX_BELT", 12)
    types = getattr(SY, "ORB_TYPES", None)
    if not types:
        n_ids = S.TYPE_BY_NAME["orb"].cat_cols[0][1] - 1
        per_orb = math.log2(max(2, n_ids)) + _range_bits("orb", "passiveValue") + _range_bits("orb", "evokeValue")
        return math.log2(cap + 1) + (cap / 2.0) * per_orb
    per_type_bits = []
    for (plo, phi), (elo, ehi) in types.values():
        per_type_bits.append(math.log2(max(2, phi - plo + 1)) + math.log2(max(2, ehi - elo + 1)))
    w = getattr(SY, "ORB_WILDCARD_PROB", 0.05)
    per_orb = math.log2(len(types)) + sum(per_type_bits) / len(per_type_bits)
    wild = math.log2(S.TYPE_BY_NAME["orb"].cat_cols[0][1]) + _range_bits("orb", "passiveValue") + _range_bits("orb", "evokeValue")
    per_orb = (1 - w) * per_orb + w * wild
    return math.log2(cap + 1) + (cap / 2.0) * per_orb


def _hist_entropy(hist) -> float:
    p = np.asarray(hist, dtype=np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def _hist_mean(hist) -> float:
    p = np.asarray(hist, dtype=np.float64)
    return float((np.arange(len(p)) * p).sum() / p.sum())


def _card_zone_bits() -> float:
    """Approx bits for a row's per-zone count vector (n_zones chosen + which zones + per-zone counts). Same
    for the table and no-table paths — the zone-vector sampler is identical in both."""
    return _hist_entropy(SY._CARD_NZONES_HIST) + 2.5 + 3.0


def _card_wild_content_bits() -> float:
    """Fully-uniform (WILDCARD) per-row CONTENT bits: every card categorical + dynamic numeric sampled
    independently over its whole range + sparse keywords (excludes the zone vector). This is the old
    over-generating per-row cost — the reachability path replaces it with the conditional table, keeping
    only a CARD_WILDCARD_PROB slice of it."""
    from .. import tokens as T
    tspec = S.TYPE_BY_NAME["card"]
    bits = 0.0
    for col_name, vocab in tspec.cat_cols:
        hi = vocab - 1 if col_name in ("type", "rarity", "targetType") else vocab
        bits += math.log2(max(2, hi))
    zone_cols = set(T.ZONE_COUNT_FIELDS)
    for c in T.CARD_NUM:
        if c in zone_cols:
            continue
        b = _range_bits("card", c)
        bits += b if b else 1.0                         # flag columns: 1 bit
    p_kw = 0.05                                          # generator keyword on-prob
    h_kw = -(p_kw * math.log2(p_kw) + (1 - p_kw) * math.log2(1 - p_kw))
    bits += T.KW_BUCKETS * h_kw
    return bits


def _dist_bits(probs) -> float:
    """Shannon entropy (bits) of a probability vector (0 for a deterministic single value)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def entropy_cards() -> float:
    """REACHABILITY-SHAPED generator (when data/reachable_v1.json exists): row count from the measured
    hist, then per row an identity from the observed cardIndex set + that card's OWN conditional value bits
    (observed type/rarity/targetType choices, enchant/afflict frequency entropy, keyword-pattern choice,
    per dynamic numeric log2 of its margin-widened observed range), a CARD_WILDCARD_PROB mix with the old
    fully-uniform content, plus the (unchanged) zone-vector bits. Falls back to the pure independent-uniform
    formula when the table is absent (the old over-generation tripwire value). Structure mirrors
    entropy_orbs()."""
    tbl = SY._try_load_reachable()
    zone_bits = _card_zone_bits()
    if tbl is None:
        per_row = _card_wild_content_bits() + zone_bits
        return _hist_entropy(SY._CARD_ROWS_HIST) + _hist_mean(SY._CARD_ROWS_HIST) * per_row
    cards = tbl["cards"]
    per_id = []
    for e in cards.values():
        b = (math.log2(max(1, len(e["type"]))) + math.log2(max(1, len(e["rarity"])))
             + math.log2(max(1, len(e["targetType"]))))
        b += _dist_bits(e["enchant_p"]) + _dist_bits(e["afflict_p"])
        b += math.log2(max(1, len(e["keywords"])))
        for _, col, _is_raw in SY._CARD_NONZONE_NUM:
            lo, hi = e["num"].get(col, (0, 0))
            b += math.log2(max(1, SY.reach_bins("card", col, lo, hi)))
        per_id.append(b)
    content = math.log2(max(2, len(cards))) + (sum(per_id) / len(per_id) if per_id else 0.0)
    w = SY.CARD_WILDCARD_PROB
    per_row = (1 - w) * content + w * _card_wild_content_bits() + zone_bits
    return _hist_entropy(SY._CARD_ROWS_HIST) + _hist_mean(SY._CARD_ROWS_HIST) * per_row


def _creature_wild_bits() -> float:
    """Fully-uniform per-creature content bits (kind enum + identity vocab + all numerics) — the WILDCARD
    creature cost the reachability path keeps only a CREATURE_WILDCARD_PROB slice of."""
    from .. import tokens as T
    cr = S.TYPE_BY_NAME["creature"]
    bits = math.log2(cr.cat_cols[0][1] - 1) + math.log2(cr.cat_cols[1][1])
    for c in T.CREATURE_NUM:
        b = _range_bits("creature", c)
        bits += b if b else 1.0
    return bits


def _counts_mean(tbl, name: str, floor1: bool = False) -> float:
    vals, probs = tbl["counts"][name]
    m = float((np.asarray(vals) * np.asarray(probs)).sum())
    return max(1.0, m) if floor1 else m


def _counts_bits(tbl, name: str) -> float:
    return _dist_bits(tbl["counts"][name][1])


def entropy_creatures() -> float:
    """REACHABILITY-SHAPED generator (when the table exists): creatures + folded powers + intents, each
    from its observed conditional table, with count TERMS driven by the measured per-state / per-creature
    histograms (replacing the old uniform 0..MAX caps that dominated the estimate). Per creature: observed
    identity + kind + margin-widened numerics; per power: observed powerIndex + amount range + parent bits;
    per intent: observed type + numerics + parent bits. A CREATURE_WILDCARD_PROB slice keeps the old
    uniform cost. Falls back to the independent-uniform formula when the table is absent. Mirrors
    entropy_orbs()."""
    from .. import tokens as T
    tbl = SY._try_load_reachable()
    if tbl is None:
        cr = S.TYPE_BY_NAME["creature"]
        per_cr = _creature_wild_bits()
        e_c = (1 + T.MAX_CREATURES) / 2.0
        pw = S.TYPE_BY_NAME["power"]
        per_pw = math.log2(pw.cat_cols[0][1]) + math.log2(e_c) + _range_bits("power", "amount")
        inn = S.TYPE_BY_NAME["intent"]
        per_in = math.log2(inn.cat_cols[0][1] - 1) + math.log2(e_c)
        for c in T.INTENT_NUM:
            b = _range_bits("intent", c)
            per_in += b if b else 1.0
        return (math.log2(T.MAX_CREATURES) + e_c * per_cr
                + math.log2(T.MAX_POWERS + 1) + (T.MAX_POWERS / 2.0) * per_pw
                + math.log2(T.MAX_INTENTS + 1) + (T.MAX_INTENTS / 2.0) * per_in)

    w = SY.CREATURE_WILDCARD_PROB
    e_c = _counts_mean(tbl, "creatures_per_state", floor1=True)
    parent_bits = math.log2(max(2, e_c))
    # Creatures.
    per_id = []
    for ce in tbl["creatures"].values():
        b = math.log2(max(1, len(ce["kind"])))
        for col in T.CREATURE_NUM:
            lo, hi = ce["num"].get(col, (0, 0))
            b += math.log2(max(1, SY.reach_bins("creature", col, lo, hi)))
        per_id.append(b)
    cr_content = math.log2(max(2, len(tbl["creatures"]))) + (sum(per_id) / len(per_id) if per_id else 0.0)
    per_cr = (1 - w) * cr_content + w * _creature_wild_bits()
    # Powers (parent adds log2(E[creatures]) bits in both paths).
    ps = tbl["powers"]
    amt_bits = (sum(math.log2(max(1, SY.reach_bins("power", "amount", lo, hi)))
                    for lo, hi in ps.values()) / len(ps)) if ps else 0.0
    pw_content = math.log2(max(2, len(ps))) + amt_bits
    pw_wild = math.log2(S.TYPE_BY_NAME["power"].cat_cols[0][1]) + _range_bits("power", "amount")
    per_pw = (1 - w) * pw_content + w * pw_wild + parent_bits
    # Intents.
    per_in_id = []
    for ie in tbl["intents"].values():
        b = 0.0
        for col in T.INTENT_NUM:
            lo, hi = ie.get(col, (0, 0))
            b += math.log2(max(1, SY.reach_bins("intent", col, lo, hi)))
        per_in_id.append(b)
    in_content = (math.log2(max(2, len(tbl["intents"])))
                  + (sum(per_in_id) / len(per_in_id) if per_in_id else 0.0))
    inn = S.TYPE_BY_NAME["intent"]
    in_wild = math.log2(inn.cat_cols[0][1] - 1)
    for c in T.INTENT_NUM:
        b = _range_bits("intent", c)
        in_wild += b if b else 1.0
    per_in = (1 - w) * in_content + w * in_wild + parent_bits
    return (_counts_bits(tbl, "creatures_per_state") + e_c * per_cr
            + _counts_bits(tbl, "powers_per_state") + _counts_mean(tbl, "powers_per_state") * per_pw
            + _counts_bits(tbl, "intents_per_state") + _counts_mean(tbl, "intents_per_state") * per_in)


ENTROPY_FNS = {"potions": entropy_potions, "relics": entropy_relics, "orbs": entropy_orbs,
               "cards": entropy_cards, "creatures": entropy_creatures}


# --------------------------------------------------------------------------------------------------
# Monte-Carlo per-STATE bits accounting.
#
# WHY this exists (learned the hard way): the analytic entropies above are MEAN bits/state. But a
# SimNorm slice reconstructs a state EXACTLY only when that state's own information cost fits the slice
# — the sizing statistic is the state-bits DISTRIBUTION's high quantiles, not its mean. The orbs
# generator has mean H=66 bits yet a 12-orb belt costs ~125 bits (~250 when several 5% wildcard rows
# land in it); a 48-bit slice plateaued at exact-coverage ~0.345 and a 96-bit slice at ~0.34 — both
# consistent with "only the states that fit decode exactly". P(state-bits <= capacity) is therefore an
# analytic ceiling on achievable exact coverage.
#
# These samplers draw n states' STRUCTURAL choices from the SAME distributions the synth fillers use
# (counts from the same histograms/uniforms, identities from the same reachable table / vocabs,
# wildcard flips with the same probs) and sum each sampled branch's information cost in bits. They do
# NOT run the generators or allocate batch arrays — it is bits arithmetic over sampled counts/identities
# only, so a 20k-state draw runs in milliseconds. Each per-branch/per-identity cost reuses the exact
# formula the matching entropy_* estimator uses (so the sample mean tracks the analytic mean), while the
# per-state SUM exposes the tail the mean hides.
# --------------------------------------------------------------------------------------------------


def _group_index(counts: np.ndarray) -> np.ndarray:
    """Within-group running index (0..k-1) for a ragged concatenation of ``counts`` groups, vectorized.
    Mirrors the position a filler's inner loop is at (used e.g. for relic duplicate targets)."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total, dtype=np.int64) - starts


def _state_bits_potions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror :func:`synth._fill_potions`: belt size k~U(0..cap); per slot a 1-bit empty/id flag (the
    generator's per-belt p_empty~U(0,1) makes the empty marginal 0.5 — exactly entropy_potions' h_slot),
    plus log2(#real ids) when the slot holds a potion. Multiset reduction ignored (as entropy_potions)."""
    cap = SY.POTION_MAX_BELT
    id_bits = math.log2(S.TYPE_BY_NAME["potion"].cat_cols[0][1] - 1)   # real ids 1..N-1
    k = rng.integers(0, cap + 1, size=n)
    bits = np.full(n, math.log2(cap + 1), dtype=np.float64)
    total = int(k.sum())
    if total:
        state_idx = np.repeat(np.arange(n), k)
        p_empty = rng.random(n)
        real = rng.random(total) >= np.repeat(p_empty, k)             # slot holds a real potion
        np.add.at(bits, state_idx, 1.0 + real.astype(np.float64) * id_bits)
    return bits


def _state_bits_relics(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror :func:`synth._fill_relics`: count k~U(0..cap); each instance is a fresh id (log2(#ids)) or,
    with RELIC_DUP_PROB, a duplicate of one already held (log2(#held) — a cheap back-reference). Matches
    entropy_relics' log2(#ids)/relic in the common case; the rare dup tail is slightly cheaper."""
    cap = SY.RELIC_MAX_SET
    id_bits = math.log2(S.TYPE_BY_NAME["relic"].cat_cols[0][1] - 1)
    k = rng.integers(0, cap + 1, size=n)
    bits = np.full(n, math.log2(cap + 1), dtype=np.float64)
    total = int(k.sum())
    if total:
        state_idx = np.repeat(np.arange(n), k)
        pos = _group_index(k)                                         # #relics already held at this row
        is_dup = (pos > 0) & (rng.random(total) < SY.RELIC_DUP_PROB)
        per = np.where(is_dup, np.log2(np.maximum(1, pos)), id_bits)
        np.add.at(bits, state_idx, per)
    return bits


def _state_bits_orbs(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror :func:`synth._fill_orbs`: belt size k~U(0..cap); per orb a 5% ORB_WILDCARD_PROB flip —
    wildcard costs log2(vocab)+full passive+full evoke range bits, otherwise log2(#types) plus THAT
    sampled type's (passive,evoke) range bits. The per-type range spread (PLASMA ~2 bits vs DARK ~11) is
    exactly the tail source: a 12-orb belt of DARK/wildcard rows dwarfs the mean-per-orb of ~10.4 bits."""
    cap = min(getattr(SY, "ORB_MAX_BELT", 12), tokens_ref().MAX_ORBS)
    types = getattr(SY, "ORB_TYPES", None) or {}
    per_type = np.array([math.log2(max(2, phi - plo + 1)) + math.log2(max(2, ehi - elo + 1))
                         for (plo, phi), (elo, ehi) in types.values()], dtype=np.float64)
    n_types = len(per_type)
    type_choice = math.log2(max(2, n_types))
    wild_bits = (math.log2(S.TYPE_BY_NAME["orb"].cat_cols[0][1])
                 + _range_bits("orb", "passiveValue") + _range_bits("orb", "evokeValue"))
    w = getattr(SY, "ORB_WILDCARD_PROB", 0.05)
    k = rng.integers(0, cap + 1, size=n)
    bits = np.full(n, math.log2(cap + 1), dtype=np.float64)
    total = int(k.sum())
    if total:
        state_idx = np.repeat(np.arange(n), k)
        is_wild = rng.random(total) < w
        tsel = rng.integers(0, n_types, size=total)
        per = np.where(is_wild, wild_bits, type_choice + per_type[tsel])
        np.add.at(bits, state_idx, per)
    return bits


def _card_id_bits(e: Dict) -> float:
    """Per-identity CONTENT bits for one card row (table path) — the identical term entropy_cards averages
    over identities. The Monte-Carlo sampler draws WHICH identity, so per-card content varies row to row."""
    b = (math.log2(max(1, len(e["type"]))) + math.log2(max(1, len(e["rarity"])))
         + math.log2(max(1, len(e["targetType"]))))
    b += _dist_bits(e["enchant_p"]) + _dist_bits(e["afflict_p"])
    b += math.log2(max(1, len(e["keywords"])))
    for _, col, _is_raw in SY._CARD_NONZONE_NUM:
        lo, hi = e["num"].get(col, (0, 0))
        b += math.log2(max(1, SY.reach_bins("card", col, lo, hi)))
    return b


def _state_bits_cards(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror :func:`synth._fill_cards`: row count from the measured rows/state histogram; each row is a
    CARD_WILDCARD_PROB fully-uniform draw or a table-conditioned identity (log2(#ids) + that id's content
    bits), plus the constant zone-vector bits. Reuses entropy_cards' exact per-identity/zone/wildcard
    costs. Falls back to the constant per-row uniform cost when the reachable table is absent."""
    hist = SY._CARD_ROWS_HIST
    cap = tokens_ref().MAX_CARDS
    count_term = _hist_entropy(hist)
    zone_bits = _card_zone_bits()
    k = np.clip(SY._sample_from_hist(rng, hist, n), 0, cap)
    tbl = SY._try_load_reachable()
    if tbl is None:
        return count_term + k.astype(np.float64) * (_card_wild_content_bits() + zone_bits)
    bits = np.full(n, count_term, dtype=np.float64)
    total = int(k.sum())
    if total == 0:
        return bits
    card_ids = tbl["card_ids"]
    per_id = np.array([_card_id_bits(tbl["cards"][int(c)]) for c in card_ids], dtype=np.float64)
    id_cost = math.log2(max(2, len(card_ids)))
    wild_bits = _card_wild_content_bits()
    state_idx = np.repeat(np.arange(n), k)
    wild = rng.random(total) < SY.CARD_WILDCARD_PROB
    sel = rng.integers(0, len(card_ids), size=total)
    content = np.where(wild, wild_bits, id_cost + per_id[sel])
    np.add.at(bits, state_idx, content + zone_bits)
    return bits


def _creature_id_bits(ce: Dict) -> float:
    """Per-identity content bits for one creature (kind choice + its numerics' reachability bins)."""
    T = tokens_ref()
    b = math.log2(max(1, len(ce["kind"])))
    for col in T.CREATURE_NUM:
        lo, hi = ce["num"].get(col, (0, 0))
        b += math.log2(max(1, SY.reach_bins("creature", col, lo, hi)))
    return b


def _intent_id_bits(ie: Dict) -> float:
    """Per-identity content bits for one intent (its numerics' reachability bins)."""
    T = tokens_ref()
    b = 0.0
    for col in T.INTENT_NUM:
        lo, hi = ie.get(col, (0, 0))
        b += math.log2(max(1, SY.reach_bins("intent", col, lo, hi)))
    return b


def _state_bits_creatures_notable(rng: np.random.Generator, n: int) -> np.ndarray:
    """No-table fallback: the independent-uniform sampling that entropy_creatures' no-table branch sums —
    c~U(1..MAX_CREATURES), powers~U(0..MAX_POWERS), intents~U(0..MAX_INTENTS), constant per-row content,
    parent bits = log2(#creatures)."""
    T = tokens_ref()
    per_cr = _creature_wild_bits()
    pw = S.TYPE_BY_NAME["power"]
    per_pw = math.log2(pw.cat_cols[0][1]) + _range_bits("power", "amount")
    inn = S.TYPE_BY_NAME["intent"]
    per_in = math.log2(inn.cat_cols[0][1] - 1)
    for c in T.INTENT_NUM:
        b = _range_bits("intent", c)
        per_in += b if b else 1.0
    c = rng.integers(1, T.MAX_CREATURES + 1, size=n).astype(np.float64)
    n_pw = rng.integers(0, T.MAX_POWERS + 1, size=n).astype(np.float64)
    n_in = rng.integers(0, T.MAX_INTENTS + 1, size=n).astype(np.float64)
    parent = np.log2(np.maximum(2.0, c))
    return (math.log2(T.MAX_CREATURES) + c * per_cr
            + math.log2(T.MAX_POWERS + 1) + n_pw * (per_pw + parent)
            + math.log2(T.MAX_INTENTS + 1) + n_in * (per_in + parent))


def _state_bits_creatures(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror :func:`synth._fill_creatures`: creature count from the measured per-state histogram (>=1);
    per creature/power/intent a CREATURE_WILDCARD_PROB uniform flip or a table-conditioned identity, with
    powers counted per-creature (histogram, capped to MAX_POWERS/state) and intents per-state (capped).
    Each power/intent adds log2(#creatures) parent bits. Reuses entropy_creatures' exact per-identity and
    wildcard costs; the tail comes from states that draw many creatures with wide-numeric identities."""
    T = tokens_ref()
    tbl = SY._try_load_reachable()
    if tbl is None:
        return _state_bits_creatures_notable(rng, n)
    w = SY.CREATURE_WILDCARD_PROB
    creature_ids, power_ids, intent_ids = tbl["creature_ids"], tbl["power_ids"], tbl["intent_ids"]
    cr_per_id = np.array([_creature_id_bits(tbl["creatures"][int(i)]) for i in creature_ids])
    pw_per_id = np.array([math.log2(max(1, SY.reach_bins("power", "amount", *tbl["powers"][int(i)])))
                          for i in power_ids])
    in_per_id = np.array([_intent_id_bits(tbl["intents"][int(i)]) for i in intent_ids])
    cr_id_cost = math.log2(max(2, len(creature_ids)))
    pw_id_cost = math.log2(max(2, len(power_ids)))
    in_id_cost = math.log2(max(2, len(intent_ids)))
    cr_wild = _creature_wild_bits()
    pw_wild = math.log2(S.TYPE_BY_NAME["power"].cat_cols[0][1]) + _range_bits("power", "amount")
    inn = S.TYPE_BY_NAME["intent"]
    in_wild = math.log2(inn.cat_cols[0][1] - 1)
    for c in T.INTENT_NUM:
        b = _range_bits("intent", c)
        in_wild += b if b else 1.0

    cvals, cprobs = tbl["counts"]["creatures_per_state"]
    c = np.clip(rng.choice(cvals, size=n, p=cprobs), 1, T.MAX_CREATURES)
    bits = np.full(n, _counts_bits(tbl, "creatures_per_state") + _counts_bits(tbl, "powers_per_state")
                   + _counts_bits(tbl, "intents_per_state"), dtype=np.float64)
    # Creatures.
    total_c = int(c.sum())
    c_state = np.repeat(np.arange(n), c)
    cw = rng.random(total_c) < w
    csel = rng.integers(0, len(creature_ids), size=total_c)
    np.add.at(bits, c_state, np.where(cw, cr_wild, cr_id_cost + cr_per_id[csel]))
    # Powers: per-creature histogram draw, summed per state, capped to MAX_POWERS (as the filler caps).
    pvals, pprobs = tbl["counts"]["powers_per_creature"]
    n_pw_state = np.zeros(n, dtype=np.int64)
    np.add.at(n_pw_state, c_state, rng.choice(pvals, size=total_c, p=pprobs))
    n_pw_state = np.minimum(n_pw_state, T.MAX_POWERS)
    total_pw = int(n_pw_state.sum())
    if total_pw:
        pw_state = np.repeat(np.arange(n), n_pw_state)
        pw_w = rng.random(total_pw) < w
        pwsel = rng.integers(0, len(power_ids), size=total_pw)
        content = np.where(pw_w, pw_wild, pw_id_cost + pw_per_id[pwsel])
        np.add.at(bits, pw_state, content + np.log2(np.maximum(2, c[pw_state])))
    # Intents: per-state histogram draw, capped to MAX_INTENTS; each references one of the c creatures.
    ivals, iprobs = tbl["counts"]["intents_per_state"]
    ni = np.minimum(rng.choice(ivals, size=n, p=iprobs), T.MAX_INTENTS)
    total_in = int(ni.sum())
    if total_in:
        in_state = np.repeat(np.arange(n), ni)
        in_w = rng.random(total_in) < w
        insel = rng.integers(0, len(intent_ids), size=total_in)
        content = np.where(in_w, in_wild, in_id_cost + in_per_id[insel])
        np.add.at(bits, in_state, content + np.log2(np.maximum(2, c[in_state])))
    return bits


_STATE_BITS_FNS = {
    "potions": _state_bits_potions, "relics": _state_bits_relics, "orbs": _state_bits_orbs,
    "cards": _state_bits_cards, "creatures": _state_bits_creatures,
}


def tokens_ref():
    """Lazy handle to the tokenizer module (avoids a top-level import; matches the local-import style the
    entropy estimators already use)."""
    from .. import tokens as T
    return T


def state_bits_sample(expert: str, n: int = 20000, seed: int = 0) -> np.ndarray:
    """``[n]`` float64 array of per-STATE information costs (bits) for ``expert``, drawn from the synth
    generator's structural distributions (see the section header). Deterministic in ``seed``. Never runs
    the generators — pure bits arithmetic over sampled counts/identities, so it returns in milliseconds."""
    fn = _STATE_BITS_FNS.get(expert)
    if fn is None:
        raise KeyError(f"state_bits_sample: unknown expert {expert!r}; known {sorted(_STATE_BITS_FNS)}")
    return fn(np.random.default_rng(seed), int(n))


# --------------------------------------------------------------------------------------------------
# Latent capacity tripwire: a SimNorm slice of width W with group g carries ~ (W/g)*log2(g) robust
# bits. An expert whose designed task entropy exceeds ~70% of that budget CANNOT reach exact coverage
# reconstruction regardless of training — flag it by arithmetic before burning GPU-hours (found live:
# orbs at H=66 vs 48-bit slice; cards/creatures generators at 2-6x their slices before reshaping).
# --------------------------------------------------------------------------------------------------

CAPACITY_WARN_FRAC = 0.7


def capacity_bits(expert: str, simnorm_group: int = 8) -> Optional[float]:
    try:
        from .model_factored import DEFAULT_SLICE_WIDTHS
    except Exception:
        return None
    w = DEFAULT_SLICE_WIDTHS.get(expert)
    return None if w is None else (w / simnorm_group) * math.log2(simnorm_group)


def capacity_report(n: int = 20000, seed: int = 0) -> List[str]:
    """Per-expert sizing table. Columns: mean analytic H (the old statistic), then the Monte-Carlo
    per-STATE bits DISTRIBUTION (p50/p95/p99/max — the statistic that actually governs exact
    reconstruction), the slice capacity, a verdict keyed on p99 vs capacity, and the estimated
    exact-coverage CEILING P(state-bits <= capacity) — the fraction of states that CAN decode exactly no
    matter how long training runs. Returns a list of lines (CLI prints them)."""
    lines = [f"{'expert':<11}{'H bits':>8}{'p50':>7}{'p95':>7}{'p99':>7}{'max':>7}"
             f"{'cap':>7}  {'verdict':<7}{'exact-ceiling':>14}"]
    for name, fn in ENTROPY_FNS.items():
        h = fn()
        cap = capacity_bits(name)
        if cap is None:
            continue
        s = state_bits_sample(name, n=n, seed=seed)
        p50, p95, p99 = (float(x) for x in np.percentile(s, [50, 95, 99]))
        ceiling = float(np.mean(s <= cap))
        verdict = ("ok" if p99 <= CAPACITY_WARN_FRAC * cap else "NEAR" if p99 <= cap else "OVER")
        lines.append(f"{name:<11}{h:>8.1f}{p50:>7.1f}{p95:>7.1f}{p99:>7.1f}{float(s.max()):>7.1f}"
                     f"{cap:>7.0f}  {verdict:<7}{ceiling * 100:>13.1f}%")
    return lines


def expert_entropy(name: str) -> Optional[float]:
    fn = ENTROPY_FNS.get(name)
    return fn() if fn else None


# --------------------------------------------------------------------------------------------------
# Milestones from run metrics.
# --------------------------------------------------------------------------------------------------

def _series(run_dir: str, metric: str) -> List[Tuple[int, float]]:
    pts = []
    for line in open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("name") == metric:
            pts.append((e["step"], e["value"]))
    pts.sort()
    return pts


def steps_to(pts: List[Tuple[int, float]], threshold: float, ascending: bool) -> Optional[int]:
    """First step at which the series crosses ``threshold`` (>= if ascending, <= otherwise)."""
    for s, v in pts:
        if (v >= threshold) if ascending else (v <= threshold):
            return s
    return None


def find_run(run_root: str, label: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(run_root, f"*{label}")))
    return hits[-1] if hits else None


# --------------------------------------------------------------------------------------------------
# Fit + verdicts.
# --------------------------------------------------------------------------------------------------

def fit_power(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Fit steps = a * H^b over (H, steps) points (log-log LSQ). Needs >= 2 points."""
    pts = [(h, s) for h, s in points if h and s]
    if len(pts) < 2:
        return None
    lh = np.log([p[0] for p in pts])
    ls = np.log([p[1] for p in pts])
    b, la = np.polyfit(lh, ls, 1)
    return float(np.exp(la)), float(b)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Entropy-scaled convergence sanity check (f(H)=a*H^b)")
    ap.add_argument("--run-dir", default="checkpoints/runs")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="run labels (suffix match), one per expert, e.g. wp5-potions-synth ...")
    ap.add_argument("--capacity-only", action="store_true",
                    help="print the capacity/state-bits report and exit (no --runs / metrics needed)")
    ap.add_argument("--experts", nargs="+", default=None,
                    help="expert name per run (default: inferred from the label)")
    ap.add_argument("--metric", default="dist", choices=["dist", "exact"],
                    help="coverage metric family: dist (descending) or exact (ascending)")
    ap.add_argument("--thresholds", default=None,
                    help="comma list; defaults: dist -> 0.2,0.1,0.05 / exact -> 0.25,0.5,0.9")
    ap.add_argument("--lag-factor", type=float, default=2.0,
                    help="actual/predicted ratio above which an expert is flagged LAGGING")
    args = ap.parse_args(argv)

    print("-- latent capacity vs designed task entropy --")
    for ln in capacity_report():
        print(ln)
    print()

    if args.capacity_only:
        return 0
    if not args.runs:
        ap.error("--runs is required unless --capacity-only is set")

    ascending = args.metric == "exact"
    metric_name = "eval.expert_exact_cov" if ascending else "eval.expert_dist_cov"
    ths = [float(x) for x in (args.thresholds.split(",") if args.thresholds
                              else (["0.25", "0.5", "0.9"] if ascending else ["0.2", "0.1", "0.05"]))]

    rows = []
    for i, label in enumerate(args.runs):
        expert = (args.experts[i] if args.experts else
                  next((e for e in ENTROPY_FNS if e in label), None))
        d = find_run(args.run_dir, label)
        if d is None or expert is None:
            print(f"!! {label}: run or expert name not resolved; skipped")
            continue
        pts = _series(d, metric_name)
        h = expert_entropy(expert)
        row = {"label": label, "expert": expert, "H": h,
               "trained": pts[-1][0] if pts else 0, "last": pts[-1][1] if pts else None,
               "cross": {t: steps_to(pts, t, ascending) for t in ths}}
        rows.append(row)

    print(f"metric={metric_name}  thresholds={ths}")
    print(f"{'expert':<10}{'H bits':>8}{'trained':>9}{'last':>8}"
          + "".join(f"{'@' + str(t):>9}" for t in ths))
    for r in rows:
        last = "-" if r["last"] is None else "%.3f" % r["last"]
        cross_cols = "".join("%9s" % (r["cross"][t] if r["cross"][t] else "-") for t in ths)
        print("%-10s%8.1f%9d%8s%s" % (r["expert"], r["H"], r["trained"], last, cross_cols))

    print()
    for t in ths:
        pts = [(r["H"], r["cross"][t]) for r in rows if r["cross"][t]]
        fit = fit_power(pts)
        if fit is None:
            print(f"@{t}: <2 crossings, no fit yet")
            continue
        a, b = fit
        print(f"@{t}: steps ~= {a:.3g} * H^{b:.2f}   (fit on {len(pts)} experts)")
        for r in rows:
            pred = a * (r["H"] ** b)
            actual = r["cross"][t]
            if actual:
                ratio = actual / pred
                flag = "LAGGING" if ratio > args.lag_factor else ("fast" if ratio < 1 / args.lag_factor else "on-curve")
                print(f"    {r['expert']:<10} actual {actual:>7}  predicted {pred:>9.0f}  x{ratio:.2f}  {flag}")
            else:
                verdict = "LAGGING (past prediction, not crossed)" if r["trained"] > args.lag_factor * pred \
                    else f"pending (predicted ~{pred:.0f}, trained {r['trained']})"
                print(f"    {r['expert']:<10} not crossed — {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
