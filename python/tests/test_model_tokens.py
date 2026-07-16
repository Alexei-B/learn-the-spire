"""Unit tests for the token set-transformer actor-critic (:mod:`lts2_agent.model_tokens`).

Synthetic states/options only — no C# host, CPU. Covers: forward shapes + masking, always-legal
sampling, finite entropy, the targeted-option -> creature-slot mapping, card-embedding sharing with the
tokenizer, checkpoint version-stamp rejection, and served-policy parity + out-of-combat decline.
"""

from __future__ import annotations

import numpy as np
import torch

from lts2_agent import model_tokens as mt
from lts2_agent import tokens
from lts2_agent.policies import torch_tokens_policy


def _card(cid="StrikeIronclad", **kw):
    c = {"cardId": cid, "energyCost": kw.get("energyCost", 1), "costsX": False,
         "type": kw.get("type", "Attack"), "rarity": "Basic",
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": False, "poolId": "X",
         "canPlay": True, "starCost": 0, "replayCount": 0, "addedKeywords": kw.get("addedKeywords", [])}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon"):
        if k in kw:
            c[k] = kw[k]
    return c


def _enemy(combat_id, hp=20):
    return {"combatId": combat_id, "monsterId": "JawWorm", "currentHp": hp, "maxHp": hp + 10,
            "block": 0, "isHittable": True, "powers": [{"powerId": "StrengthPower", "amount": 2}],
            "intents": [{"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 2}]}


def _state(enemies, osty=None, potions=None):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 1, "phase": "Play",
          "hand": [_card(damage=6, baseDamage=6)], "drawPile": [], "discardPile": [], "exhaustPile": [],
          "powers": [], "orbs": [], "orbSlots": 0, "osty": osty}
    pl = {"netId": 1, "character": "IRONCLAD", "currentHp": 50, "maxHp": 60, "block": 0, "gold": 0,
          "maxEnergy": 3, "deck": [], "relics": [], "potions": potions or [], "combatState": cs}
    return {"phase": "Combat", "seed": "s", "actIndex": 0, "floor": 1, "ascensionLevel": 0,
            "isGameOver": False, "isVictory": False, "score": 0, "players": [pl],
            "combat": {"roundNumber": 1, "currentSide": "Player", "enemies": enemies}}


def test_featurize_shapes_and_keys():
    st = _state([_enemy(10), _enemy(11)])
    opts = [{"kind": "PlayCard", "card": _card(damage=6, baseDamage=6), "targetCombatId": 10},
            {"kind": "EndTurn"}]
    f = mt.featurize(st, opts)
    for k in mt.MODEL_KEYS:
        assert k in f, k
    assert f["opt_kind"].shape == (mt.MAX_OPTIONS,)
    assert f["opt_card_idx"].shape == (mt.MAX_OPTIONS, len(tokens.CARD_IDX))
    assert f["card_idx"].shape == (tokens.MAX_CARDS, len(tokens.CARD_IDX))
    # two legal combat options, rest padded off.
    assert int(f["opt_mask"].sum()) == 2


def test_forward_shapes_and_masking():
    st1 = _state([_enemy(10), _enemy(11)])
    st2 = _state([_enemy(20)])
    opts1 = [{"kind": "PlayCard", "card": _card(), "targetCombatId": 10},
             {"kind": "PlayCard", "card": _card(), "targetCombatId": 11},
             {"kind": "EndTurn"}]
    opts2 = [{"kind": "PlayCard", "card": _card(), "targetCombatId": 20}, {"kind": "EndTurn"}]
    batch = mt.stack([mt.featurize(st1, opts1), mt.featurize(st2, opts2)])
    m = mt.TokenActorCritic(d_model=64, n_latents=4)
    logits, value = m(**mt.to_tensors(batch, "cpu"))
    assert logits.shape == (2, mt.MAX_OPTIONS)
    assert value.shape == (2,)
    # illegal (padded) options are driven to NEG_INF.
    assert bool(torch.all(logits[0, 3:] == mt.NEG_INF))
    assert bool(torch.all(logits[1, 2:] == mt.NEG_INF))
    # value head is tanh-bounded.
    assert float(value.abs().max()) <= m.value_scale


def test_targeted_option_maps_to_correct_creature_slot():
    # player -> slot 0, osty -> slot 1, enemies -> slots 2,3 (tokenizer creature order).
    st = _state([_enemy(42), _enemy(43)], osty={"currentHp": 10, "maxHp": 10, "block": 0,
                                                "isAlive": True, "powers": []})
    opts = [{"kind": "PlayCard", "card": _card(), "targetCombatId": 42},
            {"kind": "PlayCard", "card": _card(), "targetCombatId": 43},
            {"kind": "EndTurn"}]
    f = mt.featurize(st, opts)
    assert f["opt_target_slot"][0] == 2   # enemy 42
    assert f["opt_target_slot"][1] == 3   # enemy 43
    assert f["opt_target_slot"][2] == -1  # EndTurn has no target

    # Without osty, enemies shift up one slot.
    st2 = _state([_enemy(42), _enemy(43)])
    f2 = mt.featurize(st2, opts)
    assert f2["opt_target_slot"][0] == 1
    assert f2["opt_target_slot"][1] == 2


def test_option_card_matches_tokenizer_card_featurization():
    # The option's card featurizes byte-for-byte like the same card as a hand token (shared contract).
    card = _card(cid="Bash", damage=8, baseDamage=8, addedKeywords=["Retain"])
    st = _state([_enemy(10)])
    st["players"][0]["combatState"]["hand"] = [card]
    tok = tokens.tokenize(st)
    hand_row = None
    for i in range(tokens.MAX_CARDS):
        if tok["card_mask"][i]:
            hand_row = i  # only one card in hand
    f = mt.featurize(st, [{"kind": "PlayCard", "card": card, "targetCombatId": 10}])
    assert np.array_equal(f["opt_card_idx"][0], tok["card_idx"][hand_row])
    assert np.array_equal(f["opt_card_num"][0], tok["card_num"][hand_row])
    assert np.array_equal(f["opt_card_kw"][0], tok["card_kw"][hand_row])


def test_potion_option_encoded():
    st = _state([_enemy(10)], potions=["ATTACK_POTION"])
    opts = [{"kind": "UsePotion", "potionId": "ATTACK_POTION", "potionSlot": 0, "targetCombatId": 10},
            {"kind": "DiscardPotion", "potionId": "ATTACK_POTION", "potionSlot": 0},
            {"kind": "EndTurn"}]
    f = mt.featurize(st, opts)
    assert f["opt_potion_present"][0] == 1.0 and f["opt_potion_present"][1] == 1.0
    assert f["opt_potion_present"][2] == 0.0   # EndTurn: no potion
    assert f["opt_card_present"][0] == 0.0     # potion options carry no card
    assert int(f["opt_mask"][:3].sum()) == 3   # all three are combat kinds


def test_sampled_action_is_always_legal_and_entropy_finite():
    st = _state([_enemy(10), _enemy(11)])
    opts = [{"kind": "PlayCard", "card": _card(), "targetCombatId": 10}, {"kind": "EndTurn"}]
    single = _state([_enemy(10)])
    single_opts = [{"kind": "EndTurn"}]
    batch = mt.stack([mt.featurize(st, opts), mt.featurize(single, single_opts)])
    m = mt.TokenActorCritic(d_model=64, n_latents=4)
    logits, _ = m(**mt.to_tensors(batch, "cpu"))
    n_legal = [2, 1]
    for _ in range(50):
        a, lp = mt.sample_action(logits)
        for i, k in enumerate(n_legal):
            assert int(a[i]) < k, f"sampled illegal action {int(a[i])} with only {k} legal"
        assert bool(torch.isfinite(lp).all())
    ent = mt.entropy(logits)
    assert bool(torch.isfinite(ent).all()) and bool((ent >= -1e-6).all())
    assert float(ent[1]) < 1e-4   # a single legal option has ~zero entropy


def test_checkpoint_roundtrip_and_version_stamp_rejection(tmp_path):
    m = mt.TokenActorCritic(d_model=64, n_latents=4)
    ckpt = str(tmp_path / "tok.pt")
    mt.save_checkpoint(ckpt, m)
    m2, meta = mt.load_checkpoint(ckpt, "cpu")
    assert meta["tokenizer_signature"] == tokens.tokenizer_signature()
    # A checkpoint with a mismatched signature is rejected loudly (like FEATURE_VERSION).
    import json
    with open(ckpt + ".meta.json") as f:
        bad = json.load(f)
    bad["tokenizer_signature"] = "tok-v999|bogus"
    with open(ckpt + ".meta.json", "w") as f:
        json.dump(bad, f)
    try:
        mt.load_checkpoint(ckpt, "cpu")
        assert False, "expected a signature-mismatch rejection"
    except ValueError as e:
        assert "different tokenizer" in str(e)


def test_serve_parity_and_out_of_combat_decline(tmp_path):
    m = mt.TokenActorCritic(d_model=64, n_latents=4)
    ckpt = str(tmp_path / "tok.pt")
    mt.save_checkpoint(ckpt, m)
    policy = torch_tokens_policy.make_policy(ckpt)
    # Out of combat -> decline.
    assert policy({"phase": "Map"}, [{"kind": "MoveTo"}]) == []
    # In combat (greedy) -> a full ranking that agrees with the model's own argmax.
    import os
    os.environ["LTS2_POLICY_GREEDY"] = "1"
    try:
        greedy_policy = torch_tokens_policy.make_policy(ckpt)
        st = _state([_enemy(10), _enemy(11)])
        opts = [{"kind": "PlayCard", "card": _card(), "targetCombatId": 10},
                {"kind": "PlayCard", "card": _card(), "targetCombatId": 11}, {"kind": "EndTurn"}]
        logits, _ = m(**mt.to_tensors(mt.stack([mt.featurize(st, opts)]), "cpu"))
        direct = int(np.argmax(logits[0].detach().numpy()[:len(opts)]))
        ranking = greedy_policy(st, opts)
        served = max(ranking, key=lambda p: p[1])[0]
        assert served == direct
    finally:
        del os.environ["LTS2_POLICY_GREEDY"]
