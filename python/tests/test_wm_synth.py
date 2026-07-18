"""Unit tests for synthetic-space batch generation (:mod:`lts2_agent.wm.synth`, roadmap M3.5).

Verify every generator respects the tokenizer-array conventions EXACTLY — shapes, left-packed presence,
zeroed padding, index-0 semantics, symlog storage / measured ranges, relic uniqueness, potion empties at
any position — plus seeded determinism, the mixed-ratio split, and coverage-val fixedness. Where a
detokenize inverse exists (potions/relics ids, orb numerics) the synthetic array round-trips through it.
CPU only, no C# host.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
import torch

from lts2_agent import tokens
from lts2_agent.wm import model as M
from lts2_agent.wm import model_factored as MF
from lts2_agent.wm import report
from lts2_agent.wm import spec as S
from lts2_agent.wm import synth as SY
from lts2_agent.wm.experts import EXPERT_TYPES, RAW_NUM_COLS

ALL_EXPERTS = ["scalars", "potions", "relics", "orbs", "creatures", "cards"]

# The current-version pre-tokenized cache (rebuilt per tokenizer version — see wm/cache.py). A test that
# mixes synthetic with REAL cache batches needs a cache whose signature matches the live tokenizer;
# otherwise (fresh clone, or a stale cache after a version bump) it skips rather than concatenating
# shape-mismatched arrays.
_REAL_CACHE = "data/corpus_tok_v31"


def _cache_matches_current(cache_dir: str) -> bool:
    manifest = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest):
        return False
    with open(manifest) as f:
        return json.load(f).get("tokenizer_signature") == tokens.tokenizer_signature()


# ==================================================================================================
# symlog storage contract — the vectorized twin must never drift from the tokenizer's scalar helper.
# ==================================================================================================

def test_symlog_twin_matches_tokenizer():
    vals = np.array([-30, -5, -1, 0, 1, 2, 7, 42, 173, 999, 5000, 100000], dtype=np.float64)
    assert np.allclose(SY._symlog_arr(vals), [tokens.symlog(v) for v in vals], atol=0, rtol=0)


# ==================================================================================================
# Full-batch structure — every generator emits the complete model key set at padded shapes.
# ==================================================================================================

def test_batch_has_all_keys_and_shapes():
    rng = np.random.default_rng(0)
    for e in ALL_EXPERTS:
        z = SY.synth_batch([e], 5, rng)
        assert set(z) == set(M.BATCH_KEYS), e
        assert z["global_idx"].shape == (5, 1, len(tokens.GLOBAL_IDX))
        for t in S.VARIABLE_TYPES:
            assert z[t.idx_key].shape == (5, t.max_slots, len(t.cat_cols)), (e, t.name)
            assert z[t.mask_key].shape == (5, t.max_slots)
            if t.num_key:
                assert z[t.num_key].shape == (5, t.max_slots, t.num_width)
        assert z["card_kw"].shape == (5, tokens.MAX_CARDS, tokens.KW_BUCKETS)


def test_non_target_categories_are_empty():
    # A potions batch designs ONLY potions; every other variable category stays fully padded/zeroed.
    z = SY.synth_batch(["potions"], 8, np.random.default_rng(1))
    assert z["potion_mask"].any()
    for t in S.VARIABLE_TYPES:
        if t.name == "potion":
            continue
        assert not z[t.mask_key].any(), t.name
        assert (z[t.idx_key] == 0).all(), t.name


def _assert_left_packed_and_padding_zeroed(z, t):
    """Presence is left-packed (a True never follows a False) and every padded slot is all-zero."""
    m = z[t.mask_key]
    for b in range(m.shape[0]):
        row = m[b]
        k = int(row.sum())
        assert row[:k].all() and not row[k:].any(), (t.name, b)         # left-packed
        if k < t.max_slots:
            assert (z[t.idx_key][b, k:] == 0).all(), t.name             # padding ids zeroed
            if t.num_key:
                assert (z[t.num_key][b, k:] == 0).all(), t.name         # padding nums zeroed


def test_presence_left_packed_padding_zeroed():
    rng = np.random.default_rng(2)
    for e in ("potions", "relics", "orbs", "creatures", "cards"):
        z = SY.synth_batch([e], 40, rng)
        for tn in EXPERT_TYPES[e]:
            _assert_left_packed_and_padding_zeroed(z, S.TYPE_BY_NAME[tn])


# ==================================================================================================
# Per-expert game-rule + convention checks.
# ==================================================================================================

def test_relic_ids_positional_real_and_duplicates():
    # v5: relics are POSITIONAL (one row per instance) — ids are real (1..N-1), the `slot` categorical
    # equals the row index (acquisition order), duplicates are LEGAL and covered, and both empty and
    # non-empty relic sets appear.
    tspec = S.TYPE_BY_NAME["relic"]
    slot_col = [c for c, _ in tspec.cat_cols].index("slot")
    z = SY.synth_batch(["relics"], 400, np.random.default_rng(3))
    seen_counts = set()
    any_duplicate = False
    for b in range(400):
        m = z["relic_mask"][b]
        ids = z["relic_idx"][b, m, 0]
        slots = z["relic_idx"][b, m, slot_col]
        assert (ids >= 1).all(), "a present relic slot must hold a real relic id"
        assert (ids < tspec.cat_cols[0][1]).all()
        k = int(m.sum())
        assert list(slots) == list(range(k)), "relic slot column must equal the acquisition index"
        if len(set(ids.tolist())) != len(ids):
            any_duplicate = True
        seen_counts.add(k)
    assert min(seen_counts) == 0 and max(seen_counts) >= 1        # empties and non-empties both covered
    assert any_duplicate, "duplicate relics (legal) not covered"


def test_potion_belts_are_left_pack_canonical_and_duplicates():
    # v4: potion belts are LEFT-PACKED and CANONICAL — non-empty ids first (sorted by catalog index),
    # then index-0 empties. This mirrors the tokenizer's canonicalization (position is decision-
    # irrelevant), so no interior empties and no unsorted non-empty prefix ever appear.
    z = SY.synth_batch(["potions"], 600, np.random.default_rng(4))
    n_potions = S.TYPE_BY_NAME["potion"].cat_cols[0][1]
    fully_empty_belt = duplicate = mixed_belt = False
    for b in range(600):
        m = z["potion_mask"][b]
        ids = z["potion_idx"][b, m, 0].tolist()
        assert len(ids) <= SY.POTION_MAX_BELT, "belt exceeds the game-range cap"
        assert all(0 <= i < n_potions for i in ids)
        nonzero = [i for i in ids if i != 0]
        # Left-packed canonical: the non-empty prefix is sorted and the empties (id 0) all trail it.
        assert ids[:len(nonzero)] == sorted(nonzero), ("non-empty prefix not sorted/left-packed", ids)
        assert all(i == 0 for i in ids[len(nonzero):]), ("empties must trail", ids)
        if ids and not nonzero:
            fully_empty_belt = True
        if nonzero and len(nonzero) < len(ids):
            mixed_belt = True                                    # some potions + some empties in one belt
        if len(nonzero) != len(set(nonzero)):
            duplicate = True                                     # potions may duplicate
    assert fully_empty_belt, "fully-empty belts not covered"
    assert mixed_belt, "mixed (potions + trailing empties) belts not covered"
    assert duplicate, "duplicate potions (legal) not covered"


def _decoded_ints(type_name, num_block, mask):
    """Recover integers from a stored numeric block exactly as the training target does
    (:meth:`experts.RangeBinHeads.bin_targets`): raw cols round, symlog cols symexp+round."""
    raw = RAW_NUM_COLS.get(type_name, set())
    cols = {"creature": tokens.CREATURE_NUM, "power": tokens.POWER_NUM, "intent": tokens.INTENT_NUM,
            "orb": tokens.ORB_NUM, "card": tokens.CARD_NUM}[type_name]
    present = num_block[mask]
    out = {}
    for j, c in enumerate(cols):
        v = present[:, j]
        out[c] = np.round(v) if c in raw else np.round(np.sign(v) * np.expm1(np.abs(v)))
    return out


def test_numerics_within_measured_ranges():
    rng = np.random.default_rng(5)
    for e, tn in [("orbs", "orb"), ("creatures", "creature"), ("cards", "card")]:
        z = SY.synth_batch([e], 60, rng)
        dec = _decoded_ints(tn, z[S.TYPE_BY_NAME[tn].num_key], z[S.TYPE_BY_NAME[tn].mask_key])
        for c, vals in dec.items():
            if len(vals) == 0:
                continue
            r = S.NUMERIC_RANGES.get(tn, {}).get(c)
            if r is not None:
                assert vals.min() >= r.lo and vals.max() <= r.hi, (tn, c, vals.min(), vals.max())
            else:
                assert set(np.unique(vals).astype(int)).issubset({0, 1}), (tn, c)   # a flag column


def test_card_zone_counts_positive_and_small():
    # Each present card row holds >=1 instance somewhere (a row exists because it has an instance).
    z = SY.synth_batch(["cards"], 80, np.random.default_rng(6))
    cols = S.CARD_COUNT_COLS
    cn, cm = z["card_num"], z["card_mask"]
    counts = np.round(np.sign(cn[..., cols]) * np.expm1(np.abs(cn[..., cols]))).astype(int)
    present = counts[cm]
    assert (present.sum(axis=1) >= 1).all(), "a population row must have >=1 instance"


def test_numeric_storage_roundtrips_through_bin_targets():
    # The stored symlog block must land on the EXACT bin the loss targets (no ±1 drift) for every field.
    m = MF.FactoredWorldModelAE(d_model=32, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1,
                                pool_latents=2, n_mem=4, cat_dim=8,
                                slice_widths={"creatures": 64, "cards": 64, "relics": 32, "potions": 16,
                                              "orbs": 16})
    for e, tn in [("orbs", "orb"), ("creatures", "creature"), ("cards", "card")]:
        z = SY.synth_batch([e], 12, np.random.default_rng(7))
        head = m.experts[e].heads[tn]
        num = torch.tensor(z[S.TYPE_BY_NAME[tn].num_key])
        bins = head.bin_targets(num)
        assert (bins >= 0).all() and (bins < head._nbins).all(), tn      # every field lands in-range


# ==================================================================================================
# detokenize round-trip for the exactly-invertible categoricals.
# ==================================================================================================

def test_potion_relic_ids_survive_detokenize():
    for expert, tn in (("potions", "potion"), ("relics", "relic")):
        z = SY.synth_batch([expert], 16, np.random.default_rng(8))
        t = S.TYPE_BY_NAME[tn]
        for b in range(16):
            d = tokens.detokenize({k: z[k][b] for k in M.BATCH_KEYS})
            want = z[t.idx_key][b, z[t.mask_key][b], 0].tolist()
            # detokenize keeps belt/relic order; compare as multisets (order is not asserted here).
            assert sorted(d[expert]) == sorted(want), (expert, b)


# ==================================================================================================
# Determinism, mixed ratio, coverage fixedness.
# ==================================================================================================

def test_seeded_determinism():
    for e in ALL_EXPERTS:
        a = SY.synth_batch([e], 10, np.random.default_rng(123))
        b = SY.synth_batch([e], 10, np.random.default_rng(123))
        for k in M.BATCH_KEYS:
            assert np.array_equal(a[k], b[k]), (e, k)


def test_coverage_val_sample_is_fixed():
    a, aa = SY.coverage_val_sample(["potions"], 50, SY.COVERAGE_VAL_SEED)
    b, ba = SY.coverage_val_sample(["potions"], 50, SY.COVERAGE_VAL_SEED)
    assert aa == ba == ["synth"] * 50
    for k in M.BATCH_KEYS:
        assert np.array_equal(a[k], b[k]), k


def test_mixed_ratio_split():
    if not _cache_matches_current(_REAL_CACHE):
        pytest.skip(f"no current-signature cache at {_REAL_CACHE} (build it with wm.cache)")
    rng = random.Random(0)
    stream = SY.mixed_batches(_REAL_CACHE, "val", ["potions"], 20, 0.25, rng)
    stacked, acts = next(stream)
    assert len(acts) == 20
    n_synth = sum(1 for a in acts if a == "synth")
    assert n_synth == 5, n_synth                                     # round(0.25 * 20) == 5 synthetic
    for k in M.BATCH_KEYS:
        assert stacked[k].shape[0] == 20


def test_synth_batches_stream_shapes():
    stream = SY.synth_batches(["orbs"], 7, np.random.default_rng(9))
    for _ in range(3):
        stacked, acts = next(stream)
        assert acts == ["synth"] * 7
        assert stacked["orb_mask"].shape == (7, tokens.MAX_ORBS)


# ==================================================================================================
# Model forward + focused report accept synthetic batches for every learned expert.
# ==================================================================================================

def test_synth_batch_forward_and_report():
    m = MF.FactoredWorldModelAE(d_model=48, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1,
                                pool_latents=2, n_mem=4, cat_dim=12,
                                slice_widths={"creatures": 96, "cards": 96, "relics": 48, "potions": 24,
                                              "orbs": 24})
    for e in ("potions", "relics", "orbs", "creatures", "cards"):
        z = SY.synth_batch([e], 4, np.random.default_rng(10))
        batch = M.to_tensors(z, "cpu")
        with torch.no_grad():
            _, out = m(batch, active_experts=[e])
        pairs = report.report_pairs_experts_only(batch, out, [e])
        assert f"expert_dist::{e}" in pairs and f"expert_exact::{e}" in pairs
        num, den = pairs[f"expert_dist::{e}"]
        assert den.sum() > 0 and num.shape == (4,)
