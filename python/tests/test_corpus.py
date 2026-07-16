"""Corpus store + report pure-logic tests: shard round-trip, the leak-proof split rule, the seed and
explicit-deckSpec guards, and report aggregation over a synthetic in-memory corpus.

No C# host is spawned here (matching the other unit tests) — the end-to-end collection path is
verified by running the CLI manually against the built host.
"""

from __future__ import annotations

import pytest

from lts2_agent import corpus, corpus_report


# --------------------------------------------------------------------------------------------------
# Split rule: determinism + disjointness.
# --------------------------------------------------------------------------------------------------


def test_split_is_deterministic():
    seeds = [f"CORPUS-demo-{i}-{j}" for i in range(4) for j in range(50)]
    first = {s: corpus.split_for_seed(s) for s in seeds}
    for s in seeds:
        assert corpus.split_for_seed(s) == first[s]   # stable across calls
        assert first[s] in corpus.SPLITS


def test_split_buckets_partition_disjointly():
    # A fixed seed list can never land in two splits: split is a pure function of the seed.
    seeds = [f"CORPUS-x-{i}-{j}" for i in range(8) for j in range(40)]
    assigned = {}
    for s in seeds:
        assigned.setdefault(corpus.split_for_seed(s), set()).add(s)
    train, val, test = (assigned.get(k, set()) for k in corpus.SPLITS)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(seeds)


def test_split_bucket_boundaries():
    # The documented boundaries: 0-89 train, 90-94 val, 95-99 test.
    assert corpus.split_for_seed  # smoke
    # Find seeds hitting each region by scanning (crc32 is well-spread).
    seen = set()
    i = 0
    while len(seen) < 3 and i < 100000:
        seed = f"probe-region-{i}"
        b = corpus.split_bucket(seed)
        if b <= 89:
            seen.add(("train", corpus.split_for_seed(seed) == "train"))
        elif b <= 94:
            seen.add(("val", corpus.split_for_seed(seed) == "val"))
        else:
            seen.add(("test", corpus.split_for_seed(seed) == "test"))
        i += 1
    assert all(ok for _, ok in seen)


# --------------------------------------------------------------------------------------------------
# Record building + guards.
# --------------------------------------------------------------------------------------------------


def _state(hp=70, block=0, enemy_hps=(20,)):
    return {
        "phase": "Combat",
        "players": [{"character": "Ironclad", "currentHp": hp, "block": block,
                     "deck": [{"cardId": "StrikeIronclad"}, {"cardId": "DefendIronclad"}]}],
        "combat": {"enemies": [{"combatId": i, "currentHp": h} for i, h in enumerate(enemy_hps)]},
    }


def _record(seed="CORPUS-demo-0-0", regime="realistic", deck_spec="realistic",
            removed=None, added=None, won=True, done=True, t=0, character="Ironclad"):
    meta = corpus.make_scenario_meta(
        deck_spec=deck_spec, removed_cards=removed if removed is not None else ["StrikeIronclad"],
        added_cards=added if added is not None else ["Bash"], act=1, room="Monster",
        character=character, encounter="ENC", policy="heuristic", regime=regime)
    return corpus.make_record(
        seed=seed, scenario_meta=meta, t=t, state=_state(hp=70), options=[{"kind": "EndTurn"}],
        action_taken=0, next_state=_state(hp=64), next_options=[], done=done,
        info={"won": won, "hpLost": 6, "act": 1, "roomType": "Monster"})


def test_record_has_contract_keys_and_reward_components():
    rec = _record()
    corpus.validate_record(rec)
    assert set(rec) == set(corpus.RECORD_KEYS)
    rc = rec["rewardComponents"]
    assert rc["currentHp"] == {"before": 70, "after": 64}
    assert rc["enemyHp"] == {"before": 20, "after": 20}
    assert set(rec["scenarioMeta"]) == set(corpus.SCENARIO_META_KEYS)


def test_action_taken_accepts_index_or_card_indices():
    idx = _record()
    idx["actionTaken"] = 3
    corpus.validate_record(idx)
    multi = _record()
    multi["actionTaken"] = [0, 2]
    corpus.validate_record(multi)
    bad = _record()
    bad["actionTaken"] = "3"
    with pytest.raises(ValueError):
        corpus.validate_record(bad)


def test_probe_seed_is_refused():
    rec = _record(seed="PROBE-00001")
    with pytest.raises(ValueError):
        corpus.assert_writable(rec)


def test_explicit_deckspec_is_refused():
    rec = _record(deck_spec="explicit")
    with pytest.raises(corpus.ExplicitDeckSpecError):
        corpus.assert_writable(rec)
    rec2 = _record(deck_spec="random", regime="explicit")
    with pytest.raises(corpus.ExplicitDeckSpecError):
        corpus.assert_writable(rec2)


# --------------------------------------------------------------------------------------------------
# Shard writer/reader round-trip.
# --------------------------------------------------------------------------------------------------


def test_writer_round_trip_and_split_routing(tmp_path):
    root = str(tmp_path / "corpus")
    # Build fights that we know land in each split by scanning seeds.
    picked = {}
    i = 0
    while len(picked) < 3 and i < 100000:
        seed = f"CORPUS-rt-{i}"
        picked.setdefault(corpus.split_for_seed(seed), seed)
        i += 1
    assert set(picked) == set(corpus.SPLITS)

    with corpus.CorpusWriter(root, run_label="rt", shard_cap=2000) as w:
        for split, seed in picked.items():
            n = w.write_fight([_record(seed=seed, t=0, done=False),
                               _record(seed=seed, t=1, done=True)])
            assert n == 2

    # Every fight's records read back from its own split, byte-for-byte content preserved.
    for split, seed in picked.items():
        recs = [r for r in corpus.iter_records(root, split) if r["seed"] == seed]
        assert len(recs) == 2
        assert [r["t"] for r in recs] == [0, 1]
    # Total record count across the corpus.
    assert sum(1 for _ in corpus.iter_records(root)) == 6


def test_writer_rolls_shards_at_cap(tmp_path):
    root = str(tmp_path / "corpus")
    seed = next(f"CORPUS-cap-{i}" for i in range(100000)
                if corpus.split_for_seed(f"CORPUS-cap-{i}") == "train")
    with corpus.CorpusWriter(root, run_label="cap", shard_cap=3) as w:
        for _ in range(4):
            w.write_fight([_record(seed=seed)])
    paths = corpus.shard_paths(root, "train")
    assert len(paths) == 2   # 4 records, cap 3 -> two shards
    assert sum(1 for _ in corpus.iter_records(root, "train")) == 4


def test_writer_rejects_bad_fight_atomically(tmp_path):
    root = str(tmp_path / "corpus")
    with corpus.CorpusWriter(root, run_label="atom") as w:
        with pytest.raises(ValueError):
            w.write_fight([_record(seed="PROBE-1")])
    # Nothing was written.
    assert sum(1 for _ in corpus.iter_records(root)) == 0


# --------------------------------------------------------------------------------------------------
# Report aggregation over a synthetic in-memory corpus.
# --------------------------------------------------------------------------------------------------


def _synthetic_corpus(root):
    """Write a small mixed corpus: realistic + broad fights, some won, various pools."""
    fights = []
    # Realistic fights with known additions across pools.
    fights.append(_record(seed="CORPUS-syn-r1", regime="realistic", deck_spec="realistic",
                          removed=["StrikeIronclad", "DefendIronclad"], added=["Bash", "Bash"],
                          won=True, character="Ironclad"))
    fights.append(_record(seed="CORPUS-syn-r2", regime="realistic", deck_spec="realistic",
                          removed=[], added=["Apparition"], won=False, character="Ironclad"))
    # Broad fight (excluded from realistic stats).
    fights.append(_record(seed="CORPUS-syn-b1", regime="broad", deck_spec="random",
                          removed=None, added=None, won=True, character="Silent"))
    with corpus.CorpusWriter(root, run_label="syn") as w:
        for rec in fights:
            w.write_fight([rec])
    return fights


def test_report_composition_and_realistic_stats(tmp_path):
    root = str(tmp_path / "corpus")
    _synthetic_corpus(root)
    card_meta = {
        "Bash": {"pool": "ironclad", "colorless": False, "curse": False, "category": "Character"},
        "Apparition": {"pool": "colorless", "colorless": True, "curse": False, "category": "Colorless"},
    }
    report = corpus_report.build_report(root, card_meta=card_meta)

    assert report["totals"]["fights"] == 3
    # Regime breakdown: 2 realistic, 1 broad.
    regime_rows = {str(r["value"]): r for r in report["composition"]["regime"]}
    assert regime_rows["realistic"]["fights"] == 2
    assert regime_rows["broad"]["fights"] == 1

    rstats = report["realistic"]
    assert rstats["fights"] == 2
    assert rstats["removalsHist"] == {0: 1, 2: 1}
    assert rstats["additionsHist"] == {1: 1, 2: 1}
    # Two Bash (own/ironclad) + one Apparition (colorless).
    assert rstats["poolCounts"]["own"] == 2
    assert rstats["poolCounts"]["colorless"] == 1
    assert rstats["totalGradedAdditions"] == 3
    assert abs(rstats["poolRealized"]["own"] - 2 / 3) < 1e-9
    top = dict(rstats["top20Added"])
    assert top["Bash"] == 2 and top["Apparition"] == 1


def test_report_determinism_and_sample(tmp_path):
    root = str(tmp_path / "corpus")
    _synthetic_corpus(root)
    report = corpus_report.build_report(root, card_meta={})
    # Every determinism row must confirm the on-disk split equals the re-derived split.
    assert report["determinism"]
    assert all(c["stable"] for c in report["determinism"])
    # Sample decks come only from realistic fights and carry the final deck ids.
    seeds = {d["seed"] for d in report["sampleDecks"]}
    assert seeds == {"CORPUS-syn-r1", "CORPUS-syn-r2"}
    assert all(d["deck"] for d in report["sampleDecks"])


def test_classify_added_card_buckets():
    meta = {
        "Own": {"pool": "ironclad", "category": "Character"},
        "Off": {"pool": "silent", "category": "Character"},
        "Cl": {"pool": "colorless", "colorless": True, "category": "Colorless"},
        "Cu": {"pool": "curse", "curse": True, "category": "Curse"},
        "St": {"pool": "status", "status": True, "category": "Status"},
    }
    assert corpus_report.classify_added_card("Own", "Ironclad", meta) == "own"
    assert corpus_report.classify_added_card("Off", "Ironclad", meta) == "offCharacter"
    assert corpus_report.classify_added_card("Cl", "Ironclad", meta) == "colorless"
    assert corpus_report.classify_added_card("Cu", "Ironclad", meta) == "curse"
    assert corpus_report.classify_added_card("St", "Ironclad", meta) == "other"
    assert corpus_report.classify_added_card("Own", None, meta) == "ownOrOff"
    assert corpus_report.classify_added_card("missing", "Ironclad", meta) == "unknown"
