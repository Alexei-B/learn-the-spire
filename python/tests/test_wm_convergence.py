"""Unit tests for the Monte-Carlo per-STATE bits accounting in :mod:`lts2_agent.wm.convergence`.

WHY these exist: the capacity tripwire moved from MEAN bits/state (which under-sizes slices — exact
reconstruction needs the state-bits DISTRIBUTION's high quantiles) to a sampled per-state distribution
and its P(bits <= capacity) ceiling. These checks pin the sampler's contract (finite/positive, seeded,
length n), the tail ordering that motivated the change (orbs p99 >> mean), CDF monotonicity, the
finite-space experts' ~full ceilings, and hermetic operation via an injected reachable-table fixture (no
dependence on the gitignored data/reachable_v1.json). CPU only, milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from lts2_agent.wm import convergence as C
from lts2_agent.wm import synth as SY

ALL_EXPERTS = ["potions", "relics", "orbs", "cards",
               "creature-stats", "creature-powers", "creature-intents"]


# ==================================================================================================
# Reachable-table fixture — the cards/creatures samplers read SY._try_load_reachable() exactly as the
# entropy_* estimators do. Tests MUST NOT depend on the (gitignored) real artifact, so inject a tiny
# parsed fixture into the module cache for every test (copied shape from test_wm_synth.py).
# ==================================================================================================

def _fixture_card_num():
    d = {}
    for _, col, _is_raw in SY._CARD_NONZONE_NUM:
        r = SY.S.NUMERIC_RANGES.get("card", {}).get(col)
        d[col] = [0, 1] if r is None else [r.lo, min(r.lo + 2, r.hi)]
    return d


def _fixture_creature_num():
    d = {}
    for col in SY.tokens.CREATURE_NUM:
        r = SY.S.NUMERIC_RANGES.get("creature", {}).get(col)
        d[col] = [0, 1] if r is None else [r.lo, min(r.lo + 2, r.hi)]
    return d


def _fixture_intent_num():
    d = {}
    for col in SY.tokens.INTENT_NUM:
        r = SY.S.NUMERIC_RANGES.get("intent", {}).get(col)
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
            # v6 cards-as-instances: instances-per-state count + the per-zone instance marginal.
            "instances_per_state": {"0": 1, "3": 4, "5": 5, "8": 3, "12": 2},
            "card_zone": {"0": 30, "1": 33, "2": 26, "3": 10, "4": 1},
        },
    }


@pytest.fixture(autouse=True)
def _inject_reachable_table(monkeypatch):
    """Inject the fixture table into synth's module cache for every test, so the reshaped cards/creatures
    samplers run WITHOUT the real data/reachable_v1.json artifact (auto-restored by monkeypatch)."""
    monkeypatch.setattr(SY, "_REACHABLE_TABLE", SY._parse_reachable(_fixture_raw()))


# ==================================================================================================
# Sampler contract: finite, positive, length n, deterministic in the seed.
# ==================================================================================================

def test_state_bits_finite_positive_length():
    for e in ALL_EXPERTS:
        s = C.state_bits_sample(e, n=3000, seed=0)
        assert s.shape == (3000,), e
        assert np.isfinite(s).all(), e
        assert (s > 0).all(), e                                      # even an empty state costs count bits


def test_state_bits_seeded_determinism():
    for e in ALL_EXPERTS:
        a = C.state_bits_sample(e, n=1500, seed=7)
        b = C.state_bits_sample(e, n=1500, seed=7)
        assert np.array_equal(a, b), e
    # a different seed must actually move the draw (guards a silently-ignored seed).
    assert not np.array_equal(C.state_bits_sample("orbs", n=1500, seed=1),
                              C.state_bits_sample("orbs", n=1500, seed=2))


def test_unknown_expert_raises():
    with pytest.raises(KeyError):
        C.state_bits_sample("nope", n=10, seed=0)


# ==================================================================================================
# The tail that motivated the change: mean bits under-sizes; the high quantile is far heavier.
# ==================================================================================================

def test_orbs_p99_exceeds_mean_entropy():
    s = C.state_bits_sample("orbs", n=40000, seed=0)
    p99 = float(np.percentile(s, 99))
    # The sample mean must track the analytic MEAN entropy (same per-branch costs)...
    assert C.entropy_orbs() * 0.85 <= s.mean() <= C.entropy_orbs() * 1.15
    # ...but the p99 (what a slice must actually fit) is far above that mean — the whole point.
    assert p99 >= C.entropy_orbs()


# ==================================================================================================
# The exact-coverage ceiling P(bits <= capacity): monotone in capacity; ~full for finite-space experts.
# ==================================================================================================

def test_orbs_ceiling_monotone_in_capacity():
    s = C.state_bits_sample("orbs", n=60000, seed=0)
    p48, p96, p144 = (float(np.mean(s <= cap)) for cap in (48, 96, 144))
    assert p48 <= p96 <= p144
    # 128-wide (48-bit) slice was observed to plateau near 0.345 — the ceiling should land in that ballpark.
    assert 0.25 <= p48 <= 0.45
    assert p144 >= 0.95                                              # 384-wide nearly clears the space


def test_potions_relics_ceilings_are_full():
    # Both finite-space experts sit comfortably under their current slice capacity, so essentially every
    # state decodes exactly (ceiling ~1.0). This is the "ok/NEAR but not coverage-limited" regime.
    for e in ("potions", "relics"):
        cap = C.capacity_bits(e)
        s = C.state_bits_sample(e, n=20000, seed=0)
        assert float(np.mean(s <= cap)) >= 0.99, (e, cap)


# ==================================================================================================
# capacity_report(): list-of-strings contract, all experts present, one row per expert + header.
# ==================================================================================================

def test_capacity_report_shape_and_content():
    lines = C.capacity_report(n=4000, seed=0)
    assert isinstance(lines, list) and all(isinstance(x, str) for x in lines)
    assert len(lines) == 1 + len(C.ENTROPY_FNS)                     # header + one row per expert
    assert "exact-ceiling" in lines[0] and "p99" in lines[0]
    for name in C.ENTROPY_FNS:
        assert any(row.startswith(name) for row in lines[1:]), name
    # Each data row must carry a verdict token and a percentage ceiling.
    body = "\n".join(lines[1:])
    assert "%" in body
    assert any(v in body for v in ("ok", "NEAR", "OVER"))
