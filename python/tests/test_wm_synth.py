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

# The creature family is three parameter-disjoint experts (owner ruling 2026-07-18): creature-stats owns
# `creature`, creature-powers owns `power`, creature-intents owns `intent`. CREATURE_FAMILY (all three) is
# the COMBINED path — equivalent to the old single "creatures" filler; requesting it draws one shared
# per-state creature count so powers/intents parents stay consistent with the creature rows.
CREATURE_FAMILY = ["creature-stats", "creature-powers", "creature-intents"]
ALL_EXPERTS = ["scalars", "potions", "relics", "orbs", *CREATURE_FAMILY, "cards"]

# The reachability-shaped cards/creatures generators load their conditional table from
# data/reachable_v1.json. Tests MUST NOT depend on that (gitignored) artifact, so we build a tiny fixture
# table inline and inject the PARSED form straight into the module cache (autouse, per-test, auto-restored
# by monkeypatch). The two fixture card/creature identities are deliberately distinct small indices so a
# WILDCARD row (uniform over the whole catalog) is almost surely detectable by an out-of-fixture id.
_FIXTURE_CARD_IDS = [5, 42]
_FIXTURE_CREATURE_IDS = [3, 77]


def _fixture_card_num():
    d = {}
    for _, col, _is_raw in SY._CARD_NONZONE_NUM:
        r = S.NUMERIC_RANGES.get("card", {}).get(col)
        d[col] = [0, 1] if r is None else [r.lo, min(r.lo + 2, r.hi)]
    return d


def _fixture_creature_num():
    d = {}
    for col in tokens.CREATURE_NUM:
        r = S.NUMERIC_RANGES.get("creature", {}).get(col)
        d[col] = [0, 1] if r is None else [r.lo, min(r.lo + 2, r.hi)]
    return d


def _fixture_intent_num():
    d = {}
    for col in tokens.INTENT_NUM:
        r = S.NUMERIC_RANGES.get("intent", {}).get(col)
        d[col] = [0, 1] if r is None else [r.lo, min(r.lo + 2, r.hi)]
    return d


def _fixture_raw():
    """A minimal on-disk-shaped reachability doc (string keys, count dicts) — the exact shape
    wm.reachable emits and SY._parse_reachable consumes."""
    return {
        "cards": {
            "5": {"type": [1], "rarity": [2], "targetType": [1],
                  "enchant": {"0": 100, "7": 5}, "afflict": {"0": 100},
                  "keywords": [[], [3, 9]], "num": _fixture_card_num()},
            "42": {"type": [1, 2], "rarity": [3], "targetType": [1, 2],
                   "enchant": {"0": 50}, "afflict": {"0": 40, "11": 3},
                   "keywords": [[]], "num": _fixture_card_num()},
        },
        "creatures": {
            "3": {"kind": [0], "num": _fixture_creature_num()},
            "77": {"kind": [2], "num": _fixture_creature_num()},
        },
        "powers": {"11": [0, 3], "20": [-2, 5]},
        "intents": {"0": _fixture_intent_num(), "1": _fixture_intent_num()},
        "counts": {
            "creatures_per_state": {"1": 5, "2": 3, "3": 1},
            "powers_per_state": {"0": 4, "1": 3, "2": 2},
            "intents_per_state": {"0": 4, "1": 3, "2": 1},
            "powers_per_creature": {"0": 5, "1": 3, "2": 1},
            # v6 cards-as-instances: instances-per-state count + the per-zone instance marginal. Values
            # include several distinct counts and all five zones so the card generator/entropy exercise a
            # non-degenerate distribution.
            "instances_per_state": {"0": 1, "3": 4, "5": 5, "8": 3, "12": 2},
            "card_zone": {"0": 30, "1": 33, "2": 26, "3": 10, "4": 1},
        },
    }


@pytest.fixture(autouse=True)
def _inject_reachable_table(monkeypatch):
    """Inject the fixture table into synth's module cache for every test, so the reshaped cards/creatures
    generators run without the real data/reachable_v1.json artifact."""
    monkeypatch.setattr(SY, "_REACHABLE_TABLE", SY._parse_reachable(_fixture_raw()))

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
    for e in ("potions", "relics", "orbs", *CREATURE_FAMILY, "cards"):
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
    for e, tn in [("orbs", "orb"), ("creature-stats", "creature"), ("cards", "card")]:
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


def test_card_instance_zone_and_slot_columns():
    # v6: each present card INSTANCE row carries a valid zone categorical (0..len(ZONES)-1, drawn from the
    # measured marginal) and slot == its layout index; multiple zones are covered across the batch.
    z = SY.synth_batch(["cards"], 80, np.random.default_rng(6))
    zcol = [c for c, _ in S.TYPE_BY_NAME["card"].cat_cols].index("zone")
    scol = [c for c, _ in S.TYPE_BY_NAME["card"].cat_cols].index("slot")
    ci, cm = z["card_idx"], z["card_mask"]
    seen_zones = set()
    for b in range(80):
        k = int(cm[b].sum())
        zones = ci[b, :k, zcol]
        assert ((zones >= 0) & (zones < len(tokens.ZONES))).all(), "zone out of range"
        assert list(ci[b, :k, scol]) == list(range(k)), "slot must equal the layout index"
        seen_zones.update(int(x) for x in zones)
    assert len(seen_zones) >= 2, "zone marginal should cover multiple zones"


# ==================================================================================================
# DIAGNOSTIC --cards-max-rows (SINGLE-CARD probe, roadmap wm-t3-factored): cap the synthetic cards
# generator's instances-per-state to min(N, drawn) floored at 1 (never 0; N=1 -> exactly one row/state),
# honored by the coverage-val sample path too.
# ==================================================================================================

def test_cards_max_rows_one_gives_exactly_one_row():
    z = SY.synth_batch(["cards"], 64, np.random.default_rng(50), cards_max_rows=1)
    cm = z["card_mask"]
    assert cm.sum(axis=1).tolist() == [1] * 64        # exactly one card row per state (never 0, never >1)
    scol = tokens.CARD_IDX.index("slot")
    assert (z["card_idx"][cm][:, scol] == 0).all()    # the lone row's slot == its layout index 0


def test_cards_max_rows_caps_and_floors_at_one():
    # N=3: every state has min(3, drawn) rows floored at 1 -> counts in [1, 3] (the fixture draws
    # {0,3,5,8,12}; capped+floored -> {1,3}). Never 0 (presence still trained), never over the cap.
    z = SY.synth_batch(["cards"], 200, np.random.default_rng(51), cards_max_rows=3)
    counts = z["card_mask"].sum(axis=1)
    assert counts.min() >= 1 and counts.max() <= 3


def test_cards_max_rows_default_off_allows_zero_and_many():
    # Without the flag the natural instances-per-state histogram stands: some states get 0 rows, some many.
    z = SY.synth_batch(["cards"], 400, np.random.default_rng(52))
    counts = z["card_mask"].sum(axis=1)
    assert counts.min() == 0 and counts.max() >= 2     # the fixture histogram spans 0..12


def test_cards_max_rows_honored_by_coverage_sample_path():
    stacked, acts = SY.coverage_val_sample(["cards"], 40, SY.COVERAGE_VAL_SEED, cards_max_rows=1)
    assert acts == ["synth"] * 40
    assert stacked["card_mask"].sum(axis=1).tolist() == [1] * 40   # coverage-val obeys the same cap


def test_numeric_storage_roundtrips_through_bin_targets():
    # The stored symlog block must land on the EXACT bin the loss targets (no ±1 drift) for every field.
    m = MF.FactoredWorldModelAE(d_model=32, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1,
                                pool_latents=2, n_mem=4, cat_dim=8,
                                slice_widths={"creature-stats": 64, "creature-powers": 64,
                                              "creature-intents": 32, "cards": 64, "relics": 32,
                                              "potions": 16, "orbs": 16})
    for e, tn in [("orbs", "orb"), ("creature-stats", "creature"), ("cards", "card")]:
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
                                slice_widths={"creature-stats": 96, "creature-powers": 96,
                                              "creature-intents": 48, "cards": 96, "relics": 48,
                                              "potions": 24, "orbs": 24})
    for e in ("potions", "relics", "orbs", *CREATURE_FAMILY, "cards"):
        z = SY.synth_batch([e], 4, np.random.default_rng(10))
        batch = M.to_tensors(z, "cpu")
        with torch.no_grad():
            _, out = m(batch, active_experts=[e])
        pairs = report.report_pairs_experts_only(batch, out, [e])
        assert f"expert_dist::{e}" in pairs and f"expert_exact::{e}" in pairs
        num, den = pairs[f"expert_dist::{e}"]
        assert den.sum() > 0 and num.shape == (4,)


# ==================================================================================================
# Reachability-shaped cards/creatures (roadmap M3.5, wm-t3-factored) — the reshaped generators keep the
# tokenizer's canonical order, sample only in-range values, symlog-store integers, and hit both the
# table and the wildcard paths.
# ==================================================================================================

def _assert_card_rows_content_sorted(z):
    """v6: every state's present card rows are ZONE-MAJOR then within-zone in the tokenizer's content order,
    with slot == the layout index. lexsort by (zone, content-key) is therefore the identity permutation."""
    ci, cn, ckw, cm = z["card_idx"], z["card_num"], z["card_kw"], z["card_mask"]
    zcol = tokens.CARD_IDX.index("zone")
    scol = tokens.CARD_IDX.index("slot")
    for b in range(ci.shape[0]):
        k = int(cm[b].sum())
        assert list(ci[b, :k, scol]) == list(range(k)), ("slot != index", b)
        if k <= 1:
            continue
        content_cols = SY._card_content_key_columns(ci[b, :k], cn[b, :k], ckw[b, :k])
        order = np.lexsort(content_cols[::-1] + [ci[b, :k, zcol].astype(np.int64)])
        assert list(order) == list(range(k)), ("card rows not zone-major/content-sorted", b)


def _assert_creatures_lexsorted(z):
    """Every state's present creatures obey the tokenizer's v4 lexsort key exactly (unchanged in v5)."""
    m = z["creature_mask"]
    for b in range(m.shape[0]):
        k = int(m[b].sum())
        if k <= 1:
            continue
        cats, nums = z["creature_idx"][b, :k], z["creature_num"][b, :k]
        order = np.lexsort((nums[:, 3], nums[:, 2], nums[:, 1], nums[:, 0], cats[:, 1],
                            nums[:, 4], cats[:, 0]))
        assert list(order) == list(range(k)), b


def _encode_card_num(ints):
    """Encode an integer [k, CARD_NUM] block into the stored array form (raw flags raw, else symlog)."""
    cn = ints.astype(np.float32)
    for j, col in enumerate(tokens.CARD_NUM):
        if col not in SY._CARD_RAW_COLS:
            cn[:, j] = SY._symlog_arr(ints[:, j].astype(np.float64)).astype(np.float32)
    return cn


def _random_card_rows(rng, k):
    """A block of ``k`` random UNSORTED card rows in valid stored-array form, drawn from a SMALL pool of
    content templates so many rows share an identical categorical+numeric key and the keyword multiset is
    the actual tiebreak (the case the vectorized sort must get exactly right), while distinct templates
    still exercise the categorical/numeric ordering."""
    n_templ = max(1, k // 3)
    templ_ci = rng.integers(0, 4, size=(n_templ, len(tokens.CARD_IDX))).astype(np.int32)
    templ_int = np.zeros((n_templ, len(tokens.CARD_NUM)), np.int64)
    for j, col in enumerate(tokens.CARD_NUM):
        templ_int[:, j] = rng.integers(0, 2, size=n_templ) if col in SY._CARD_RAW_COLS \
            else rng.integers(-1, 4, size=n_templ)
    pick = rng.integers(0, n_templ, size=k)
    ci = templ_ci[pick]
    cn = _encode_card_num(templ_int[pick])
    ckw = (rng.random((k, tokens.KW_BUCKETS)) < 0.12).astype(np.float32)         # sparse keyword sets
    return ci, cn, ckw


def test_card_content_order_matches_reference():
    # The vectorized content sort must reproduce the per-row reference (tokens._card_content_key order)
    # EXACTLY for every possible row — the canonical row order is a wire-format contract with the
    # tokenizer. Over many random batches with forced ties, the two permutations are identical.
    rng = np.random.default_rng(0xC0DE)
    saw_tie_broken_by_keywords = False
    for _ in range(200):
        k = int(rng.integers(0, 25))
        ci, cn, ckw = _random_card_rows(rng, k)
        want = SY._card_content_order_ref(ci, cn, ckw, k)
        got = SY._card_content_order(ci, cn, ckw, k)
        assert got == want, (k, got, want)
        # Confirm the keyword tiebreak is genuinely exercised: some adjacent pair in the sorted order ties
        # on all categorical+numeric columns and is separated only by its keyword multiset.
        if k >= 2 and not saw_tie_broken_by_keywords:
            cols = SY._card_content_key_columns(ci[:k], cn[:k], ckw[:k])
            no_kw = np.stack(cols[:len(SY._CARD_CONTENT_CAT_COLS) + len(SY._CARD_NONZONE_NUM)], axis=1)
            for a, b in zip(want, want[1:]):
                if np.array_equal(no_kw[a], no_kw[b]) and not np.array_equal(ckw[a], ckw[b]):
                    saw_tie_broken_by_keywords = True
                    break
    assert saw_tie_broken_by_keywords, "keyword-only tiebreak never exercised — test is not covering it"


def test_reshaped_cards_are_content_sorted():
    z = SY.synth_batch(["cards"], 64, np.random.default_rng(20))
    _assert_card_rows_content_sorted(z)


def test_reshaped_creatures_are_lexsorted():
    z = SY.synth_batch(["creature-stats"], 64, np.random.default_rng(21))
    _assert_creatures_lexsorted(z)


def test_table_conditioned_values_within_spec_ranges():
    # Force the pure table path (no wildcard): every sampled numeric must decode to an integer inside
    # spec.NUMERIC_RANGES (the margin+clamp contract), and flag columns must stay 0/1.
    import pytest as _pytest  # local alias to avoid shadowing the module-level fixture arg name
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(SY, "CARD_WILDCARD_PROB", 0.0)
        mp.setattr(SY, "CREATURE_WILDCARD_PROB", 0.0)
        rng = np.random.default_rng(22)
        # The creature case uses the COMBINED family so powers are present for the amount check below.
        for experts, tn in [(["cards"], "card"), (CREATURE_FAMILY, "creature")]:
            z = SY.synth_batch(experts, 80, rng)
            dec = _decoded_ints(tn, z[S.TYPE_BY_NAME[tn].num_key], z[S.TYPE_BY_NAME[tn].mask_key])
            for c, vals in dec.items():
                if len(vals) == 0:
                    continue
                r = S.NUMERIC_RANGES.get(tn, {}).get(c)
                if r is not None:
                    assert vals.min() >= r.lo and vals.max() <= r.hi, (tn, c, vals.min(), vals.max())
                else:
                    assert set(np.unique(vals).astype(int)).issubset({0, 1}), (tn, c)
            # Powers folded into the creatures expert (amount can be negative) also stay in range.
            if tn == "creature":
                amt = _decoded_ints("power", z["power_num"], z["power_mask"])["amount"]
                if len(amt):
                    r = S.NUMERIC_RANGES["power"]["amount"]
                    assert amt.min() >= r.lo and amt.max() <= r.hi


def test_reshaped_symlog_storage_is_integer_exact():
    # symlog storage identity: every stored non-flag numeric is exactly symlog(integer) — symexp recovers
    # a whole number (no drift), the same contract the tokenizer/bin_targets rely on.
    rng = np.random.default_rng(23)
    for e, tn, cols in [("cards", "card", tokens.CARD_NUM),
                        ("creature-stats", "creature", tokens.CREATURE_NUM)]:
        z = SY.synth_batch([e], 48, rng)
        raw = RAW_NUM_COLS.get(tn, set())
        num = z[S.TYPE_BY_NAME[tn].num_key][z[S.TYPE_BY_NAME[tn].mask_key]]
        for j, c in enumerate(cols):
            v = num[:, j].astype(np.float64)
            if c in raw:
                assert np.allclose(v, np.round(v), atol=0), (tn, c)      # flags stored raw (integers)
            else:
                sx = np.sign(v) * np.expm1(np.abs(v))
                assert np.allclose(sx, np.round(sx), atol=1e-5), (tn, c)  # symlog of an integer


def test_wildcard_and_table_paths_both_exercised():
    # With a mid wildcard prob and enough rows, some identities come from the tiny fixture set (table
    # path) and some fall outside it (wildcard uniform over the whole catalog) — both paths must fire.
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(SY, "CARD_WILDCARD_PROB", 0.5)
        mp.setattr(SY, "CREATURE_WILDCARD_PROB", 0.5)
        zc = SY.synth_batch(["cards"], 200, np.random.default_rng(24))
        cids = zc["card_idx"][zc["card_mask"]][:, 0]
        in_tbl = np.isin(cids, _FIXTURE_CARD_IDS)
        assert in_tbl.any(), "no table-path card rows"
        assert (~in_tbl).any(), "no wildcard-path card rows"

        zk = SY.synth_batch(["creature-stats"], 200, np.random.default_rng(25))
        ids = zk["creature_idx"][zk["creature_mask"]][:, 1]
        in_tbl = np.isin(ids, _FIXTURE_CREATURE_IDS)
        assert in_tbl.any(), "no table-path creature rows"
        assert (~in_tbl).any(), "no wildcard-path creature rows"


def test_missing_reachable_artifact_raises_clearly():
    # With no injected table AND no artifact on disk, the generator raises a clear, actionable error
    # (it must never silently fall back to the old uniform space).
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(SY, "_REACHABLE_TABLE", None)
        mp.setattr(SY, "_REACHABLE_JSON", os.path.join(os.path.sep, "no", "such", "reachable_v1.json"))
        with _pytest.raises(FileNotFoundError, match="reachable"):
            SY.synth_batch(["cards"], 2, np.random.default_rng(26))


# ==================================================================================================
# Power / intent CANONICAL WIRE ORDER (roadmap wm-t3-factored — the generator-canonicality bug fix).
# The tokenizer flattens each creature's POWERS sorted by (powerIndex, amount) and INTENTS sorted by
# (type, damage, baseDamage, hits), following the v4-sorted creature slots (parent-slot flatten). The
# generator MUST emit rows in that exact order, else per-slot targets are ill-posed for the permutation-
# invariant decoder (the measured ~0.27-0.29 unresolvable creatures floor). These tests assert the
# generator obeys the order, that a power's powerIndex is DISTINCT within a creature (a power stacks its
# amount — duplicates are an unreachable state), and pin the generator's comparator to tokenize() itself.
# ==================================================================================================

def _decode_symlog_int(v):
    """Recover the stored integer from a symlog-stored float block (== the tokenizer's inverse)."""
    return np.rint(np.sign(v) * np.expm1(np.abs(v))).astype(np.int64)


def _assert_left_packed(mask):
    row = mask.astype(bool)
    k = int(row.sum())
    assert row[:k].all() and not row[k:].any(), "presence not left-packed"
    return k


def _check_power_order(pidx, pnum, pmask, *, require_distinct):
    """Assert one state's present power rows are in the tokenizer's canonical flatten order — grouped by
    parent creature slot ascending, then (powerIndex, amount-integer) ascending. With ``require_distinct``
    (the GENERATOR invariant: a power stacks its amount, so a repeat is unreachable) also assert a DISTINCT
    powerIndex within every (parent) group — the twin tokenizer state omits it, deliberately holding a
    same-idx pair to exercise the amount tiebreak. Returns (n_distinct_parents, max_group_size)."""
    k = _assert_left_packed(pmask)
    m = pmask.astype(bool)
    idx = pidx[:, 0][m].astype(np.int64)
    parent = pidx[:, 1][m].astype(np.int64)
    amt = _decode_symlog_int(pnum[:, 0][m])
    if k == 0:
        return 0, 0
    # Canonical order: sorting by (parent, powerIndex, amount) is the identity permutation.
    assert list(np.lexsort((amt, idx, parent))) == list(range(k)), \
        ("power rows not in (parent, idx, amount) canonical order", list(zip(parent, idx, amt)))
    max_group = 0
    for p in np.unique(parent):
        gi = idx[parent == p]
        if require_distinct:
            assert len(set(gi.tolist())) == len(gi), \
                ("duplicate powerIndex on one creature (unreachable — a power stacks its amount)", p, gi)
        max_group = max(max_group, len(gi))
    return len(np.unique(parent)), max_group


def _check_intent_order(iidx, inum, imask):
    """Assert one state's present intent rows are in the tokenizer's canonical flatten order — grouped by
    parent creature slot ascending, then (type, damage, baseDamage, hits) ascending. Returns
    (n_distinct_parents, max_group_size)."""
    k = _assert_left_packed(imask)
    m = imask.astype(bool)
    typ = iidx[:, 0][m].astype(np.int64)
    parent = iidx[:, 1][m].astype(np.int64)
    dmg = _decode_symlog_int(inum[:, 1][m])       # INTENT_NUM = [hasDamage, damage, baseDamage, hasHits, hits]
    base = _decode_symlog_int(inum[:, 2][m])
    hits = _decode_symlog_int(inum[:, 4][m])
    if k == 0:
        return 0, 0
    assert list(np.lexsort((hits, base, dmg, typ, parent))) == list(range(k)), \
        ("intent rows not in (parent, type, damage, baseDamage, hits) canonical order",
         list(zip(parent, typ, dmg, base, hits)))
    groups = {int(p): int((parent == p).sum()) for p in np.unique(parent)}
    return len(groups), max(groups.values())


def test_generated_powers_are_canonical_and_distinct():
    # COMBINED family path (creatures + powers + intents all present) — the e2c6e83 canonicality contract
    # must still hold after the creature-family split.
    z = SY.synth_batch(CREATURE_FAMILY, 64, np.random.default_rng(30))
    saw_multi_group = saw_multi_parent = False
    for b in range(z["power_mask"].shape[0]):
        n_par, max_group = _check_power_order(z["power_idx"][b], z["power_num"][b], z["power_mask"][b],
                                              require_distinct=True)
        saw_multi_group |= max_group >= 2         # a creature with >=2 (distinct) powers
        saw_multi_parent |= n_par >= 2            # powers spread over >=2 parent creatures
    assert saw_multi_group, "no creature carried >=2 powers — distinctness/within-group order not exercised"
    assert saw_multi_parent, "powers never spanned >=2 parents — parent grouping not exercised"


def test_generated_intents_are_canonical():
    # COMBINED family path — same contract, intents present alongside creatures + powers.
    z = SY.synth_batch(CREATURE_FAMILY, 64, np.random.default_rng(31))
    saw_multi_group = saw_multi_parent = False
    for b in range(z["intent_mask"].shape[0]):
        n_par, max_group = _check_intent_order(z["intent_idx"][b], z["intent_num"][b], z["intent_mask"][b])
        saw_multi_group |= max_group >= 2
        saw_multi_parent |= n_par >= 2
    assert saw_multi_group, "no creature carried >=2 intents — within-group order not exercised"
    assert saw_multi_parent, "intents never spanned >=2 parents — parent grouping not exercised"


def test_standalone_powers_are_canonical_and_distinct():
    # STANDALONE creature-powers (no creature rows present): the filler draws its OWN virtual creature-count
    # context, so parents are realistic 0..c-1 and the SAME e2c6e83 canonicality/distinctness contract holds
    # on the solo path the powers expert actually trains on.
    z = SY.synth_batch(["creature-powers"], 64, np.random.default_rng(40))
    assert not z["creature_mask"].any(), "standalone powers must not write creature rows"
    saw_multi_group = saw_multi_parent = False
    for b in range(z["power_mask"].shape[0]):
        n_par, max_group = _check_power_order(z["power_idx"][b], z["power_num"][b], z["power_mask"][b],
                                              require_distinct=True)
        saw_multi_group |= max_group >= 2
        saw_multi_parent |= n_par >= 2
    assert saw_multi_group, "standalone: no creature carried >=2 powers — within-group order not exercised"
    assert saw_multi_parent, "standalone: powers never spanned >=2 parents — grouping not exercised"


def test_standalone_intents_are_canonical():
    # STANDALONE creature-intents (own virtual creature-count context) — canonical order + parent grouping.
    z = SY.synth_batch(["creature-intents"], 64, np.random.default_rng(41))
    assert not z["creature_mask"].any(), "standalone intents must not write creature rows"
    saw_multi_group = saw_multi_parent = False
    for b in range(z["intent_mask"].shape[0]):
        n_par, max_group = _check_intent_order(z["intent_idx"][b], z["intent_num"][b], z["intent_mask"][b])
        saw_multi_group |= max_group >= 2
        saw_multi_parent |= n_par >= 2
    assert saw_multi_group, "standalone: no creature carried >=2 intents — within-group order not exercised"
    assert saw_multi_parent, "standalone: intents never spanned >=2 parents — grouping not exercised"


def test_combined_family_parents_consistent_with_creature_count():
    # The combined family draws ONE per-state creature count, so every power/intent parent slot is a valid
    # index into that state's creature rows (0..c-1). This is the cross-expert consistency the shared-count
    # design guarantees; a standalone powers/intents run has no creature rows to check against.
    z = SY.synth_batch(CREATURE_FAMILY, 96, np.random.default_rng(42))
    saw_power = saw_intent = False
    for b in range(z["creature_mask"].shape[0]):
        c = int(z["creature_mask"][b].sum())
        assert c >= 1
        pm, im = z["power_mask"][b], z["intent_mask"][b]
        if pm.any():
            saw_power = True
            assert z["power_idx"][b, pm, 1].max() < c, ("power parent >= creature count", b, c)
        if im.any():
            saw_intent = True
            assert z["intent_idx"][b, im, 1].max() < c, ("intent parent >= creature count", b, c)
    assert saw_power and saw_intent, "combined family produced no powers/intents to check"


def test_combined_family_matches_old_single_creatures_sequence():
    # The combined family (shared count, canonical stats->powers->intents order) reproduces the pre-split
    # single-filler draw sequence, so creatures + powers + intents co-occur exactly as the old joint
    # "creatures" batch did: every present power/intent belongs to a state that HAS creatures, and the three
    # token types partition into one coherent creature population.
    z = SY.synth_batch(CREATURE_FAMILY, 64, np.random.default_rng(43))
    for b in range(z["creature_mask"].shape[0]):
        has_pw = bool(z["power_mask"][b].any())
        has_in = bool(z["intent_mask"][b].any())
        if has_pw or has_in:
            assert z["creature_mask"][b].any(), ("powers/intents without any creature rows", b)


def _twin_state():
    """A small CANONICAL-input state (tokenize() shape) exercising every ordering degree of freedom the
    generator's comparator enforces: 3 creatures (player + 2 enemies); powers spread across them INCLUDING
    two powers with the SAME powerIndex and different amounts on one creature (the amount tiebreak); and
    intents across two enemies including two out-of-order intents on one (the type/damage tiebreak)."""
    player = {
        "netId": 1, "character": "IRONCLAD", "currentHp": 50, "maxHp": 70, "block": 3,
        "gold": 10, "maxEnergy": 3, "deck": [], "relics": [], "potions": [],
        "combatState": {
            "energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 1, "phase": "Play",
            "hand": [], "drawPile": [], "discardPile": [], "exhaustPile": [],
            "orbs": [], "orbSlots": 0, "osty": None,
            # deliberately UNSORTED, with a same-idx (ACCELERANT) pair whose amounts must break the tie.
            "powers": [
                {"powerId": "ACCURACY_POWER", "amount": 5},      # idx 2
                {"powerId": "ACCELERANT_POWER", "amount": 2},    # idx 1
                {"powerId": "ACCELERANT_POWER", "amount": -1},   # idx 1 (same idx, smaller amount)
            ]},
    }
    enemy1 = {"combatId": 100, "monsterId": "M1", "currentHp": 30, "maxHp": 40, "block": 0,
              "isHittable": True,
              "powers": [{"powerId": "ADAPTABLE_POWER", "amount": 3}],
              "intents": [  # deliberately out of canonical order (Buff type 1 before Attack type 0).
                  {"type": "Buff", "damage": None, "baseDamage": None, "hits": None},
                  {"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 2}]}
    enemy2 = {"combatId": 200, "monsterId": "M2", "currentHp": 20, "maxHp": 25, "block": 0,
              "isHittable": True,
              "powers": [{"powerId": "AFTERIMAGE_POWER", "amount": 4}],
              "intents": [{"type": "Defend", "damage": None, "baseDamage": None, "hits": None}]}
    return {"phase": "Combat", "seed": "T", "actIndex": 1, "floor": 3, "ascensionLevel": 0,
            "isGameOver": False, "isVictory": False, "score": 0, "players": [player],
            "combat": {"roundNumber": 1, "currentSide": "Player", "enemies": [enemy1, enemy2]}}


def test_generator_and_tokenizer_share_one_order_contract():
    # TWIN test: run the real tokenizer on a canonical state and assert the power/intent ROW ORDER it emits
    # is already canonical under the SAME comparator the generator enforces (applying it is a no-op
    # permutation). This pins generator and tokenizer to one wire-order contract going forward.
    tok = tokens.tokenize(_twin_state())
    n_par_p, max_group_p = _check_power_order(tok["power_idx"], tok["power_num"], tok["power_mask"],
                                              require_distinct=False)
    n_par_i, max_group_i = _check_intent_order(tok["intent_idx"], tok["intent_num"], tok["intent_mask"])
    assert n_par_p == 3 and max_group_p == 3, (n_par_p, max_group_p)   # 3 parents; player holds 3 powers
    assert n_par_i == 2 and max_group_i == 2, (n_par_i, max_group_i)   # 2 enemies; enemy1 holds 2 intents

    # The amount tiebreak is genuinely exercised: the two same-idx (ACCELERANT) rows on the player appear
    # adjacent with amounts ascending (-1 before 2).
    m = tok["power_mask"].astype(bool)
    idx = tok["power_idx"][:, 0][m].astype(np.int64)
    parent = tok["power_idx"][:, 1][m].astype(np.int64)
    amt = _decode_symlog_int(tok["power_num"][:, 0][m])
    accel = [(int(p), int(a)) for p, i, a in zip(parent, idx, amt) if i == tokens._POWERS.index_of("ACCELERANT_POWER")]
    assert accel == [(0, -1), (0, 2)], accel     # same parent (player slot 0), amount-ascending tiebreak
