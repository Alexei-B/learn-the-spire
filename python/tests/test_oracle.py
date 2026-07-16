"""Oracle prober pure-logic tests: probe (de)serialization, the PROBE- seed-namespace guard, and the
shard record schema. No C# host is spawned here (matching the other unit tests) — the end-to-end
replay path is verified by running the CLI manually against the built host.
"""

from __future__ import annotations

import json
import random

import pytest

from lts2_agent import oracle
from lts2_agent.oracle import (
    PROBE_SEED_PREFIX,
    Probe,
    assert_not_probe_seed,
    is_probe_seed,
    make_shard_record,
    plan_probe,
    validate_probe_seed,
    validate_shard_record,
)


# --------------------------------------------------------------------------------------------------
# Seed namespace guard.
# --------------------------------------------------------------------------------------------------

def test_is_probe_seed():
    assert is_probe_seed("PROBE-00001")
    assert is_probe_seed(PROBE_SEED_PREFIX)
    assert not is_probe_seed("AGENT")
    assert not is_probe_seed("probe-1")  # case-sensitive on purpose


def test_validate_probe_seed_accepts_prefixed():
    assert validate_probe_seed("PROBE-42") == "PROBE-42"


def test_validate_probe_seed_rejects_unprefixed():
    with pytest.raises(ValueError):
        validate_probe_seed("RUN1")
    with pytest.raises(ValueError):
        validate_probe_seed("C0-3")


def test_assert_not_probe_seed_guards_training_seeds():
    assert assert_not_probe_seed("RUN1") == "RUN1"
    with pytest.raises(ValueError):
        assert_not_probe_seed("PROBE-00001")


def test_plan_probe_always_prefixes_and_is_deterministic():
    for i in range(50):
        rp_a, steps_a = plan_probe(random.Random(f"m:{i}"), i, None)
        rp_b, steps_b = plan_probe(random.Random(f"m:{i}"), i, None)
        # PROBE- namespace enforced for every generated probe.
        assert is_probe_seed(rp_a["seed"])
        # Same RNG seed + index -> identical plan (reproducible builds).
        assert rp_a == rp_b and steps_a == steps_b
        assert 0 <= steps_a <= 15
        assert 0 <= rp_a["act"] <= 2
        # Room class encodes as clean elite/boss pct knobs.
        assert (rp_a["elitePct"], rp_a["bossPct"]) in {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)}


def test_plan_probe_samples_given_characters():
    chars = ["Ironclad", "Necrobinder"]
    seen = set()
    for i in range(100):
        rp, _ = plan_probe(random.Random(f"c:{i}"), i, chars)
        seen.add(rp["character"])
    # Over many draws we see both a specified character and the "omit -> host random" (None) case.
    assert None in seen
    assert seen & set(chars)


# --------------------------------------------------------------------------------------------------
# Probe (de)serialization round-trip.
# --------------------------------------------------------------------------------------------------

def _sample_probe() -> Probe:
    return Probe(
        probe_id="probe-00007",
        reset_params={
            "seed": "PROBE-00007",
            "character": "Ironclad",
            "elitePct": 1.0,
            "bossPct": 0.0,
            "starterDeck": True,
            "act": 2,
        },
        action_prefix=[0, 3, 1],
        meta={"act": 2, "roomType": "Elite", "character": "IRONCLAD", "turn": 3,
              "phase": "Combat", "optionCount": 5},
    )


def test_probe_round_trip():
    p = _sample_probe()
    p2 = Probe.from_dict(p.to_dict())
    assert p2.probe_id == p.probe_id
    assert p2.reset_params == p.reset_params
    assert p2.action_prefix == p.action_prefix
    assert p2.meta == p.meta


def test_probe_to_dict_is_json_stable():
    p = _sample_probe()
    d = p.to_dict()
    # Exactly the four documented keys.
    assert set(d) == {"probeId", "resetParams", "actionPrefix", "meta"}
    # reset params carry only the reserved keys, camelCase.
    assert set(d["resetParams"]) == {"seed", "character", "elitePct", "bossPct", "starterDeck", "act"}
    # Fully JSON round-trippable.
    again = Probe.from_dict(json.loads(json.dumps(d)))
    assert again.to_dict() == d


def test_probe_rejects_non_probe_seed():
    with pytest.raises(ValueError):
        Probe(probe_id="x", reset_params={"seed": "RUN1", "elitePct": 0.0, "bossPct": 0.0},
              action_prefix=[])


def test_probe_reset_kwargs_maps_to_env_signature():
    p = _sample_probe()
    kw = p.reset_kwargs()
    assert kw == {"seed": "PROBE-00007", "character": "Ironclad", "elite_pct": 1.0,
                  "boss_pct": 0.0, "starter_deck": True, "act": 2}


def test_save_and_load_probe_set(tmp_path):
    probes = [
        _sample_probe(),
        Probe(probe_id="probe-00001", reset_params={"seed": "PROBE-00001", "elitePct": 0.0,
                                                    "bossPct": 0.0, "starterDeck": False, "act": 0},
              action_prefix=[2], meta={}),
    ]
    path = str(tmp_path / "probes.json")
    oracle.save_probe_set(path, probes, generator={"n": 2, "masterSeed": "m"})
    header, loaded = oracle.load_probe_set(path)
    assert header["count"] == 2
    assert header["generator"]["masterSeed"] == "m"
    assert "createdAt" in header
    # Sorted by probeId.
    assert [p.probe_id for p in loaded] == ["probe-00001", "probe-00007"]


# --------------------------------------------------------------------------------------------------
# Shard record schema.
# --------------------------------------------------------------------------------------------------

def _fake_obs() -> dict:
    return {"state": {"phase": "Combat"}, "options": [{"kind": "EndTurn"}], "done": False, "info": {}}


def test_full_shard_record_schema():
    rec = make_shard_record(
        "probe-1",
        position=_fake_obs(),
        results=[{"action": 0, "obs": _fake_obs()}, {"action": 1, "error": "boom"}],
        meta={"act": 1},
    )
    validate_shard_record(rec)  # must not raise
    assert rec["probeId"] == "probe-1"
    assert rec["position"]["state"]["phase"] == "Combat"
    assert rec["results"][0]["action"] == 0 and "obs" in rec["results"][0]
    assert rec["results"][1]["error"] == "boom"
    assert rec["meta"] == {"act": 1}
    # Round-trips through JSONL.
    validate_shard_record(json.loads(json.dumps(rec)))


def test_error_shard_record_schema():
    rec = make_shard_record("probe-9", position=None, error="position replay failed")
    validate_shard_record(rec)
    assert rec == {"probeId": "probe-9", "error": "position replay failed"}


def test_validate_shard_record_rejects_bad_records():
    with pytest.raises(ValueError):
        validate_shard_record({"position": {}, "results": []})  # missing probeId
    with pytest.raises(ValueError):
        validate_shard_record({"probeId": "p"})  # neither error nor position/results
    with pytest.raises(ValueError):
        validate_shard_record({"probeId": "p", "position": {}, "results": [{"obs": {}}]})  # no action
    with pytest.raises(ValueError):
        validate_shard_record({"probeId": "p", "position": {}, "results": [{"action": 0}]})  # no obs/error


# --------------------------------------------------------------------------------------------------
# Observation helpers.
# --------------------------------------------------------------------------------------------------

def test_clean_obs_strips_transport_cruft():
    obs = {"state": {"a": 1}, "options": [], "done": True, "info": {}, "_bytes": 123, "protocolVersion": 1}
    cleaned = oracle.clean_obs(obs)
    assert cleaned == {"state": {"a": 1}, "options": [], "done": True, "info": {}}


def test_capture_meta_reads_stratification_fields():
    obs = {
        "state": {"phase": "Combat", "actIndex": 1,
                  "players": [{"character": "IRONCLAD", "combatState": {"turnNumber": 4}}]},
        "options": [{"kind": "EndTurn"}, {"kind": "PlayCard"}],
        "info": {"act": 1, "roomType": "Elite"},
    }
    meta = oracle.capture_meta(obs)
    assert meta == {"act": 1, "roomType": "Elite", "character": "IRONCLAD",
                    "turn": 4, "phase": "Combat", "optionCount": 2}


def test_is_freezable():
    assert oracle._is_freezable({"done": False, "options": [1, 2], "state": {"phase": "Combat"}})
    assert oracle._is_freezable({"done": False, "options": [1, 2], "state": {"phase": "Choice"}})
    assert not oracle._is_freezable({"done": True, "options": [1, 2], "state": {"phase": "Combat"}})
    assert not oracle._is_freezable({"done": False, "options": [1], "state": {"phase": "Combat"}})
    assert not oracle._is_freezable({"done": False, "options": [1, 2], "state": {"phase": "Reward"}})
