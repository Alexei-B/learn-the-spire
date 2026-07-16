"""Unit tests for the pre-tokenized corpus cache (:mod:`lts2_agent.wm.cache`).

Small synthetic corpus written via :class:`corpus.CorpusWriter` in a tmp dir; CPU only, workers=1 (no
multiprocessing spawn). Covers: build -> load/verify round-trip byte-equality vs a fresh tokenize;
signature-mismatch rejection; and deterministic seeded shard/in-shard shuffle.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import pytest

from lts2_agent import corpus
from lts2_agent.wm import cache as C
from lts2_agent.wm import data as D
from lts2_agent.wm import model as M


# --------------------------------------------------------------------------------------------------
# Synthetic corpus builders.
# --------------------------------------------------------------------------------------------------

def _card(cid="StrikeIronclad", **kw):
    c = {"cardId": cid, "energyCost": kw.get("energyCost", 1), "costsX": False,
         "type": kw.get("type", "Attack"), "rarity": "Basic",
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": False, "poolId": "X",
         "canPlay": True, "starCost": 0, "replayCount": 0, "addedKeywords": []}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon"):
        if k in kw:
            c[k] = kw[k]
    return c


def _state(hp=50, n_hand=2, n_enemies=1):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 1, "phase": "Play",
          "hand": [_card(damage=6, baseDamage=6) for _ in range(n_hand)],
          "drawPile": [_card("Defend", block=5, baseBlock=5, type="Skill")],
          "discardPile": [], "exhaustPile": [], "powers": [], "orbs": [], "orbSlots": 0, "osty": None}
    pl = {"netId": 1, "character": "IRONCLAD", "currentHp": hp, "maxHp": 60, "block": 0, "gold": 0,
          "maxEnergy": 3, "deck": [], "relics": ["BurningBlood"], "potions": [], "combatState": cs}
    return {"phase": "Combat", "seed": "T", "actIndex": 1, "floor": 3, "ascensionLevel": 0,
            "isGameOver": False, "isVictory": False, "score": 0, "players": [pl],
            "combat": {"roundNumber": 1, "currentSide": "Player",
                       "enemies": [{"combatId": i + 1, "monsterId": "JawWorm", "currentHp": 20 - i,
                                    "maxHp": 30, "block": 0, "isHittable": True, "powers": [],
                                    "intents": [{"type": "Attack", "damage": 6, "baseDamage": 6,
                                                 "hits": 1}]} for i in range(n_enemies)]}}


def _fight_records(seed: str, act: int, n: int) -> List[Dict[str, Any]]:
    """A contiguous fight of ``n`` records; record t's nextState == record t+1's state."""
    meta = corpus.make_scenario_meta(deck_spec="broad", removed_cards=None, added_cards=None,
                                     act=act, room="Monster", character="IRONCLAD", encounter="e",
                                     policy="heuristic", regime="broad")
    states = [_state(hp=50 - i, n_hand=(i % 3) + 1, n_enemies=(i % 2) + 1) for i in range(n + 1)]
    recs = []
    for t in range(n):
        recs.append(corpus.make_record(
            seed=seed, scenario_meta=meta, t=t, state=states[t], options=[{"kind": "EndTurn"}],
            action_taken=0, next_state=states[t + 1], next_options=[{"kind": "EndTurn"}],
            done=(t == n - 1), info=None))
    return recs


def _seed_for_split(want: str, start: int = 0) -> str:
    """Find a non-probe seed that maps to ``want`` split."""
    i = start
    while True:
        s = f"fight-{i}"
        if corpus.split_for_seed(s) == want:
            return s
        i += 1


def _write_corpus(root: str) -> None:
    with corpus.CorpusWriter(root, run_label="syn", shard_cap=1000) as w:
        # A handful of fights per split so both train and val are populated.
        for split, act in (("train", 0), ("train", 1), ("val", 2), ("val", 0), ("test", 1)):
            seed = _seed_for_split(split, start=hash((split, act)) % 1000)
            w.write_fight(_fight_records(seed, act, n=5))


def _fresh_states(root: str, split: str) -> List[Dict[str, np.ndarray]]:
    out = []
    for state, _act in D.iter_states(root, split):
        f = D._featurize_safe(state)
        if f is not None:
            out.append(f)
    return out


# --------------------------------------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------------------------------------

def test_build_load_roundtrip_and_verify(tmp_path):
    root = str(tmp_path / "corpus")
    cache_dir = str(tmp_path / "corpus_tok")
    _write_corpus(root)

    manifest = C.build(root, cache_dir, workers=1, shard_size=7)
    assert manifest["dedup"] == "both-states"
    assert manifest["tokenizer_signature"]
    assert set(manifest["splits"]) == set(corpus.SPLITS)

    # Cache state count == fresh state count (both-states parity; nothing deduped or dropped).
    for split in corpus.SPLITS:
        fresh = _fresh_states(root, split)
        assert manifest["splits"][split]["n_states"] == len(fresh)

    # Cached first-N == fresh first-N, byte for byte (fixed val sample parity).
    fresh_val = _fresh_states(root, "val")
    stacked, acts = D.load_fixed_sample_from_cache(cache_dir, "val", len(fresh_val))
    assert len(acts) == len(fresh_val)
    for i, f in enumerate(fresh_val):
        for k in M.BATCH_KEYS:
            assert np.array_equal(stacked[k][i], f[k])
            assert stacked[k][i].dtype == f[k].dtype

    # The builder's own equivalence gate passes over a sample.
    checked, total = C.verify(root, cache_dir, "val", n_sample=total_states(manifest, "val"))
    assert checked == total == len(fresh_val)


def total_states(manifest, split):
    return manifest["splits"][split]["n_states"]


def test_signature_mismatch_rejects(tmp_path):
    root = str(tmp_path / "corpus")
    cache_dir = str(tmp_path / "corpus_tok")
    _write_corpus(root)
    C.build(root, cache_dir, workers=1, shard_size=7)

    # Valid cache resolves.
    assert C.resolve_manifest(cache_dir) is not None
    # Absent cache -> None (fall back to on-the-fly), no raise.
    assert C.resolve_manifest(str(tmp_path / "nope")) is None
    assert C.resolve_manifest(None) is None

    # Corrupt the stamped signature -> loud rejection (not silent fallback).
    mpath = os.path.join(cache_dir, C.MANIFEST_NAME)
    with open(mpath) as f:
        man = json.load(f)
    man["tokenizer_signature"] = "tok-vDIFFERENT"
    with open(mpath, "w") as f:
        json.dump(man, f)
    with pytest.raises(RuntimeError, match="different tokenizer"):
        C.resolve_manifest(cache_dir)
    # And the loader dispatchers propagate the loud error.
    with pytest.raises(RuntimeError, match="different tokenizer"):
        D.load_fixed_sample(root, "val", 4, cache_dir=cache_dir)


def test_seeded_shuffle_is_deterministic(tmp_path):
    root = str(tmp_path / "corpus")
    cache_dir = str(tmp_path / "corpus_tok")
    _write_corpus(root)
    C.build(root, cache_dir, workers=1, shard_size=3)  # small shards -> real shard-order shuffle

    def take(seed, n):
        # creature_num encodes per-state HP, so its order across batches reveals the shuffle order.
        gen = D.cache_batches_cpu(cache_dir, "train", batch_size=2, rng=random.Random(seed))
        return [(np.copy(b["creature_num"]), list(a)) for b, a in (next(gen) for _ in range(n))]

    a = take(123, 8)
    b = take(123, 8)
    c = take(999, 8)
    # Same seed -> identical batch sequence.
    for (ba, aa), (bb, ab) in zip(a, b):
        assert np.array_equal(ba, bb)
        assert aa == ab
    # Different seed -> the stream differs somewhere (shard/in-shard order changed).
    assert any(not np.array_equal(ba, bc) for (ba, _), (bc, _) in zip(a, c))


def test_cache_batches_cover_all_states(tmp_path):
    """Every cached state appears within one epoch's worth of batches (no drops besides the <batch
    remainder carried forward)."""
    root = str(tmp_path / "corpus")
    cache_dir = str(tmp_path / "corpus_tok")
    _write_corpus(root)
    manifest = C.build(root, cache_dir, workers=1, shard_size=4)
    n_train = manifest["splits"]["train"]["n_states"]

    gen = D.cache_batches_cpu(cache_dir, "train", batch_size=3, rng=random.Random(0))
    seen = 0
    n_batches = n_train // 3
    for _ in range(n_batches):
        b, acts = next(gen)
        assert b["global_num"].shape[0] == 3
        seen += len(acts)
    assert seen == n_batches * 3
