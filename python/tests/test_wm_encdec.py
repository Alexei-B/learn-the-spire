"""Unit tests for the world-model encoder/decoder (:mod:`lts2_agent.wm`).

Synthetic tokenized states only — no C# host, CPU. Covers: encoder/decoder forward shapes on synthetic
states, SimNorm latent normalization, loss decreases over ~50 overfit steps (gradient sanity), the report
card metric contract + reconstruction<->detokenize hand-off, and checkpoint stamp rejection.
"""

from __future__ import annotations

import numpy as np
import torch

from lts2_agent import tokens
from lts2_agent.wm import model as M
from lts2_agent.wm import report
from lts2_agent.wm import spec as S
from lts2_agent.wm.decoder import reconstruct_arrays
from lts2_agent.wm.encoder import simnorm


def _card(cid="StrikeIronclad", **kw):
    c = {"cardId": cid, "energyCost": kw.get("energyCost", 1), "costsX": False,
         "type": kw.get("type", "Attack"), "rarity": "Basic",
         "targetType": kw.get("targetType", "AnyEnemy"), "upgraded": False, "poolId": "X",
         "canPlay": True, "starCost": 0, "replayCount": 0, "addedKeywords": []}
    for k in ("damage", "baseDamage", "block", "baseBlock", "summon"):
        if k in kw:
            c[k] = kw[k]
    return c


def _enemy(combat_id=1, hp=20):
    return {"combatId": combat_id, "monsterId": "JawWorm", "currentHp": hp, "maxHp": hp + 10,
            "block": 3, "isHittable": True, "powers": [{"powerId": "StrengthPower", "amount": 2}],
            "intents": [{"type": "Attack", "damage": 6, "baseDamage": 6, "hits": 2}]}


def _state(n_hand=2, n_enemies=1, relics=None, potions=None):
    cs = {"energy": 3, "maxEnergy": 3, "stars": 0, "turnNumber": 1, "phase": "Play",
          "hand": [_card(damage=6, baseDamage=6) for _ in range(n_hand)],
          "drawPile": [_card("Defend", block=5, baseBlock=5, type="Skill")],
          "discardPile": [], "exhaustPile": [], "powers": [{"powerId": "StrengthPower", "amount": 1}],
          "orbs": [], "orbSlots": 0, "osty": None}
    pl = {"netId": 1, "character": "IRONCLAD", "currentHp": 50, "maxHp": 60, "block": 0, "gold": 0,
          "maxEnergy": 3, "deck": [], "relics": relics or ["BurningBlood"],
          "potions": potions or [], "combatState": cs}
    return {"phase": "Combat", "seed": "T", "actIndex": 1, "floor": 3, "ascensionLevel": 0,
            "isGameOver": False, "isVictory": False, "score": 0, "players": [pl],
            "combat": {"roundNumber": 1, "currentSide": "Player",
                       "enemies": [_enemy(i + 1) for i in range(n_enemies)]}}


def _batch(states, device="cpu"):
    feats = [M.featurize(s) for s in states]
    return M.to_tensors(M.collate(feats), device)


def _small_model():
    return M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                          n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8)


def test_forward_shapes():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    model = _small_model()
    z, out = model(batch)
    assert z.shape == (2, 128)
    # Per-type decoder outputs: presence for variable types, numerics/cats present, right slot counts.
    for t in S.TYPES:
        o = out[t.name]
        assert len(o["cat"]) == len(t.cat_cols)
        for c, (_, vocab) in zip(o["cat"], t.cat_cols):
            assert c.shape == (2, t.max_slots, vocab)
        if t.num_width:
            assert o["num"].shape == (2, t.max_slots, t.num_width)
        if t.mask_key:
            assert o["presence"].shape == (2, t.max_slots)


def test_simnorm_is_normalized():
    z = torch.randn(4, 128)
    zn = simnorm(z, 8)
    # Each group of 8 is a probability simplex: non-negative, sums to 1.
    groups = zn.reshape(4, -1, 8)
    assert torch.all(zn >= 0)
    assert torch.allclose(groups.sum(-1), torch.ones(4, 16), atol=1e-5)
    # And the encoder's z is normalized the same way.
    _z, _ = _small_model()(_batch([_state()]))
    g = _z.reshape(1, -1, 8)
    assert torch.allclose(g.sum(-1), torch.ones(1, 16), atol=1e-5)


def test_overfit_one_batch_decreases_loss():
    torch.manual_seed(0)
    states = [_state(n_hand=i % 4 + 1, n_enemies=i % 2 + 1) for i in range(6)]
    batch = _batch(states)
    model = _small_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = None
    last = None
    for i in range(50):
        _z, out = model(batch)
        losses = M.compute_losses(batch, out)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        if i == 0:
            first = float(losses["loss"])
        last = float(losses["loss"])
    assert last < 0.5 * first, f"loss did not drop enough: {first:.3f} -> {last:.3f}"


def test_report_card_contract_and_detokenize_handoff():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    model = _small_model()
    _z, out = model(batch)
    pairs = report.report_pairs(batch, out)
    assert set(pairs) == set(report.METRIC_NAMES)
    overall = report.aggregate(pairs)
    for name in report.METRIC_NAMES:
        assert np.isfinite(overall[name])
    # Reconstructed arrays feed tokens.detokenize verbatim (decoder output IS the tokenizer array space).
    arrays = reconstruct_arrays(out)
    canon = tokens.detokenize(arrays[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


def test_exact_state_rate_is_one_when_decoding_targets():
    # Feed the tokenizer's own target arrays as if they were perfect decoder logits: exact_state_rate=1.
    batch = _batch([_state(), _state(n_hand=2, n_enemies=2)])
    B = batch["global_idx"].shape[0]
    fake_out = {}
    for t in S.TYPES:
        o = {}
        # One-hot logits at the target index -> argmax recovers the target exactly.
        o["cat"] = []
        if t.cat_cols:
            for c, (_, vocab) in enumerate(t.cat_cols):
                oh = torch.zeros(B, t.max_slots, vocab)
                idx = batch[t.idx_key][..., c].clamp(0, vocab - 1)
                oh.scatter_(-1, idx.unsqueeze(-1), 10.0)
                o["cat"].append(oh)
        if t.num_width:
            o["num"] = batch[t.num_key].clone()
        if t.mask_key:
            o["presence"] = torch.where(batch[t.mask_key], torch.tensor(10.0), torch.tensor(-10.0))
        if t.has_kw:
            o["kw"] = torch.where(batch["card_kw"] > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
        fake_out[t.name] = o
    pairs = report.report_pairs(batch, fake_out)
    overall = report.aggregate(pairs)
    assert overall["exact_state_rate"] == 1.0
    assert overall["card_id_top1"] == 1.0
    assert overall["energy_acc"] == 1.0


def test_checkpoint_stamp_rejection(tmp_path):
    model = _small_model()
    path = str(tmp_path / "wm.pt")
    M.save_checkpoint(path, model, step=7)
    # Round-trips when the signature matches.
    loaded, meta = M.load_checkpoint(path, "cpu")
    assert meta["step"] == 7
    # Corrupt the stamped signature -> load must reject loudly.
    import json
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    meta["tokenizer_signature"] = "tok-vDIFFERENT"
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    try:
        M.load_checkpoint(path, "cpu")
        assert False, "expected a signature-mismatch rejection"
    except ValueError as e:
        assert "different tokenizer" in str(e)
