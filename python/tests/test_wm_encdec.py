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
from lts2_agent.wm.decoder import (NUM_BINS, _dedup_slot_ids, reconstruct_arrays, symlog_bins,
                                   twohot_expectation, twohot_targets)
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


def _small_tokens_model(latent_k=6):
    # tokens mode: no flatten/z_dim/n_mem — the latent is latent_k x d_model, SimNorm per token.
    return M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                          n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8,
                          latent_mode="tokens", latent_k=latent_k)


def _twohot_model():
    return M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                          n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8,
                          num_head="twohot")


def _set_relic_model():
    return M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                          n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8,
                          relic_head="set")


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


def test_tokens_mode_forward_shapes_and_per_token_simnorm():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    model = _small_tokens_model(latent_k=6)
    z, out = model(batch)
    # Latent is a token SET (no flatten): [B, latent_k, d_model].
    assert z.shape == (2, 6, 64)
    # SimNorm is applied PER latent token over its d_model channels: each group of 8 sums to 1.
    groups = z.reshape(2, 6, -1, 8)
    assert torch.all(z >= 0)
    assert torch.allclose(groups.sum(-1), torch.ones(2, 6, 8), atol=1e-5)
    # Decoder output space is UNCHANGED from flat mode (same per-type heads / slot counts).
    for t in S.TYPES:
        o = out[t.name]
        assert len(o["cat"]) == len(t.cat_cols)
        for c, (_, vocab) in zip(o["cat"], t.cat_cols):
            assert c.shape == (2, t.max_slots, vocab)
        if t.num_width:
            assert o["num"].shape == (2, t.max_slots, t.num_width)
        if t.mask_key:
            assert o["presence"].shape == (2, t.max_slots)


def test_tokens_mode_reconstruct_arrays_detokenize_handoff():
    # The new memory path must feed reconstruct_arrays -> detokenize identically to flat mode.
    batch = _batch([_state(), _state(n_hand=2, n_enemies=2)])
    model = _small_tokens_model()
    _z, out = model(batch)
    pairs = report.report_pairs(batch, out)
    assert set(pairs) == set(report.METRIC_NAMES)
    arrays = reconstruct_arrays(out)
    canon = tokens.detokenize(arrays[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


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


def test_card_spec_num_width_includes_zone_counts():
    # v3: the card numeric block carries the per-zone count vector (5 count_<zone> columns) and zone
    # left the categorical block; the spec follows tokens.CARD_* mechanically, so the decoder heads
    # widen/narrow automatically.
    card = S.TYPE_BY_NAME["card"]
    assert card.num_width == len(tokens.CARD_NUM)
    assert tokens.ZONE_COUNT_FIELDS == ["count_" + z for z in tokens.ZONES]
    assert all(f in tokens.CARD_NUM for f in tokens.ZONE_COUNT_FIELDS)
    assert [c[0] for c in card.cat_cols] == tokens.CARD_IDX  # zone no longer a categorical column
    # The decoder emits a num vector of exactly that width for the card type.
    batch = _batch([_state()])
    model = _small_model()
    _z, out = model(batch)
    assert out["card"]["num"].shape[-1] == len(tokens.CARD_NUM)


def test_grouped_cards_reduce_token_count_through_model_path():
    # A duplicate-heavy hand tokenizes to fewer card tokens than instances; the batch/model path and
    # the reconstruct->detokenize handoff still work with grouped tokens.
    st = _state(n_hand=1)
    cs = st["players"][0]["combatState"]
    cs["drawPile"] = [_card(damage=6, baseDamage=6) for _ in range(6)]  # 6 identical strikes
    tok = tokens.tokenize(st)
    n_draw_instances = 6
    n_tokens = int(tok["card_mask"].sum())
    assert n_tokens < n_draw_instances + len(cs["hand"])  # grouping happened
    batch = _batch([st])
    model = _small_model()
    _z, out = model(batch)
    arrays = reconstruct_arrays(out)
    canon = tokens.detokenize(arrays[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


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


def test_checkpoint_latent_mode_roundtrip(tmp_path):
    model = _small_tokens_model(latent_k=6)
    path = str(tmp_path / "wm_tokens.pt")
    M.save_checkpoint(path, model, step=5)
    loaded, meta = M.load_checkpoint(path, "cpu")
    # latent_mode/latent_k surface both top-level (for M4) and in config (for reconstruction).
    assert meta["latent_mode"] == "tokens"
    assert meta["latent_k"] == 6
    assert meta["config"]["latent_mode"] == "tokens"
    assert meta["config"]["latent_k"] == 6
    # The reloaded model still produces the token-set latent.
    z, _ = loaded(_batch([_state()]))
    assert z.shape[1:] == (6, 64)


def test_flat_checkpoint_defaults_latent_mode(tmp_path):
    # An existing (flat) checkpoint whose config predates latent_mode must still load as flat.
    model = _small_model()
    path = str(tmp_path / "wm_flat.pt")
    M.save_checkpoint(path, model, step=2)
    loaded, meta = M.load_checkpoint(path, "cpu")
    assert meta["latent_mode"] == "flat"
    z, _ = loaded(_batch([_state()]))
    assert z.shape == (1, 128)


def test_checkpoint_latent_mode_mismatch_rejected(tmp_path):
    model = _small_model()  # flat
    path = str(tmp_path / "wm_flat.pt")
    M.save_checkpoint(path, model, step=1)
    # Requesting the other mode on load rejects loudly (flat/tokens are not interchangeable).
    try:
        M.load_checkpoint(path, "cpu", expect_latent_mode="tokens")
        assert False, "expected a latent_mode-mismatch rejection"
    except ValueError as e:
        assert "latent_mode" in str(e)
    # Matching mode (or no expectation) loads fine.
    _loaded, meta = M.load_checkpoint(path, "cpu", expect_latent_mode="flat")
    assert meta["latent_mode"] == "flat"


# ==================================================================================================
# Probe flag 1: two-hot numeric head (--num-head twohot).
# ==================================================================================================

def test_twohot_target_roundtrip_and_shape():
    bins = symlog_bins(NUM_BINS)
    # Synthetic symlog values (incl. symlog'd integers), all inside the grid range.
    vals = torch.tensor([[0.0, 1.234, -3.5, 5.0],
                         [tokens.symlog(7), tokens.symlog(42), 0.1, -0.1]], dtype=torch.float32)
    probs = twohot_targets(vals, bins)
    assert probs.shape == (2, 4, NUM_BINS)
    # A proper two-hot: non-negative, sums to 1, at most two nonzero bins.
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 4), atol=1e-6)
    assert int((probs > 1e-8).sum(-1).max()) <= 2
    # Expectation over the bins reconstructs the original value exactly (linear-interp round-trip).
    rt = twohot_expectation(probs, bins)
    assert torch.allclose(rt, vals, atol=1e-4)


def test_twohot_head_shapes_and_downstream_handoff():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    model = _twohot_model()
    z, out = model(batch)
    for t in S.TYPES:
        o = out[t.name]
        if t.num_width:
            # Decoded value has the SAME shape/role as the MSE head (downstream unchanged) ...
            assert o["num"].shape == (2, t.max_slots, t.num_width)
            # ... plus the per-column bin logits used only by the CE loss.
            assert o["num_logits"].shape == (2, t.max_slots, t.num_width, NUM_BINS)
    # The decoded numerics feed report_pairs + reconstruct_arrays + detokenize identically to mse mode.
    pairs = report.report_pairs(batch, out)
    assert set(pairs) == set(report.METRIC_NAMES)
    canon = tokens.detokenize(reconstruct_arrays(out)[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


def test_twohot_loss_trains():
    torch.manual_seed(0)
    states = [_state(n_hand=i % 4 + 1, n_enemies=i % 2 + 1) for i in range(6)]
    batch = _batch(states)
    model = _twohot_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = last = None
    for i in range(40):
        _z, out = model(batch)
        assert "num_logits" in out["global"]      # two-hot path active
        losses = M.compute_losses(batch, out)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        if i == 0:
            first = float(losses["loss"])
        last = float(losses["loss"])
    assert last < first, f"twohot loss did not drop: {first:.3f} -> {last:.3f}"


# ==================================================================================================
# Probe flag 2: class-balanced card CE (--card-ce balanced).
# ==================================================================================================

def test_card_ce_weights_shape_and_normalization():
    n = S.TYPE_BY_NAME["card"].cat_cols[0][1]
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 1000, size=n).astype(np.int64)
    counts[0] = 0  # an unseen class must still get a finite, bounded weight
    w = M.card_ce_weights_from_counts(counts)
    assert w.shape == (n,)
    assert w.dtype == np.float32
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    # Documented normalization: the FREQUENCY-weighted mean weight is 1 (loss scale ~ unchanged).
    fw_mean = float((counts * w).sum() / counts.sum())
    assert abs(fw_mean - 1.0) < 1e-4
    # Rare classes weigh more than common ones (1/sqrt(freq)).
    assert w[int(counts.argmin())] > w[int(counts.argmax())]


def test_card_ce_plain_mode_unchanged_and_weight_only_hits_card_col():
    batch = _batch([_state(n_hand=3), _state(n_hand=2, n_enemies=2)])
    torch.manual_seed(1)
    model = _small_model()
    _z, out = model(batch)
    base = M.compute_losses(batch, out)
    # Passing weights=None is byte-identical to the default (plain) call.
    same = M.compute_losses(batch, out, card_ce_weights=None)
    assert float(same["loss"]) == float(base["loss"])
    assert float(same["loss_categorical"]) == float(base["loss_categorical"])
    # A non-uniform weight vector changes the categorical loss (via the card-id column only).
    n = S.TYPE_BY_NAME["card"].cat_cols[0][1]
    torch.manual_seed(2)
    w = torch.rand(n) + 0.5
    weighted = M.compute_losses(batch, out, card_ce_weights=w)
    assert float(weighted["loss_categorical"]) != float(base["loss_categorical"])
    # Numeric/presence terms are untouched by the card weighting.
    assert float(weighted["loss_numeric"]) == float(base["loss_numeric"])
    assert float(weighted["loss_presence"]) == float(base["loss_presence"])


# ==================================================================================================
# Probe flag 3: weight EMA (--ema DECAY).
# ==================================================================================================

def test_ema_update_math_and_val_swap():
    torch.manual_seed(0)
    model = _small_model()
    ema = M.EMA(model, decay=0.9)
    name = next(n for n, p in model.named_parameters() if torch.is_floating_point(p))
    shadow_old = ema.shadow[name].clone()
    assert torch.allclose(shadow_old, model.state_dict()[name])   # shadow starts == live weights
    # Perturb the live weights, then step the EMA.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    new = model.state_dict()[name].clone()
    ema.update(model)
    assert torch.allclose(ema.shadow[name], 0.9 * shadow_old + 0.1 * new, atol=1e-6)
    # store -> copy_to swaps the EMA weights into the live model (what the val pass evaluates) ...
    ema.store(model)
    ema.copy_to(model)
    assert torch.allclose(model.state_dict()[name], 0.9 * shadow_old + 0.1 * new, atol=1e-6)
    # ... and restore puts the training weights back untouched.
    ema.restore(model)
    assert torch.equal(model.state_dict()[name], new)


def test_ema_checkpoint_roundtrip(tmp_path):
    model = _small_model()
    ema = M.EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    path = str(tmp_path / "wm_ema.pt")
    M.save_checkpoint(path, model, step=3, extra={"ema_decay": 0.5}, ema_state=ema.state_dict())
    import os
    assert os.path.exists(path + ".ema")
    loaded, meta = M.load_checkpoint(path, "cpu")
    assert meta["ema_decay"] == 0.5
    ema2 = M.EMA(loaded, decay=0.5)
    ema2.load_state_dict(torch.load(path + ".ema", map_location="cpu"))
    name = next(n for n, p in model.named_parameters() if torch.is_floating_point(p))
    assert torch.allclose(ema2.shadow[name], ema.shadow[name], atol=1e-6)


# ==================================================================================================
# Regression guard: every flag OFF => byte-identical model + loss to the pre-flag baseline.
# ==================================================================================================

def test_num_head_default_off_is_mse_and_state_dict_identical():
    torch.manual_seed(0)
    default = _small_model()                                   # no num_head arg -> default
    torch.manual_seed(0)
    explicit = M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                              n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8,
                              num_head="mse")
    assert default.cfg["num_head"] == "mse"
    ka, kb = default.state_dict(), explicit.state_dict()
    assert ka.keys() == kb.keys()                              # no twohot-only tensors leak in
    for k in ka:
        assert ka[k].shape == kb[k].shape and torch.equal(ka[k], kb[k])
    # And the mse forward never produces bin logits.
    _z, out = default(_batch([_state()]))
    assert "num_logits" not in out["global"]
    assert "num" in out["global"]


def test_all_flags_off_loss_matches_baseline():
    # With num_head=mse and no card weights (the defaults), compute_losses is the base loss verbatim.
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    torch.manual_seed(3)
    model = _small_model()
    _z, out = model(batch)
    base = M.compute_losses(batch, out)
    again = M.compute_losses(batch, out, card_ce_weights=None)
    for k in base:
        assert float(base[k]) == float(again[k])


# ==================================================================================================
# CP4 relic-decode fixes: slot-head dedup (fix 1) + multi-hot set head (--relic-head set, fix 2).
# ==================================================================================================

def test_relic_slot_dedup_removes_duplicates_and_keeps_confident():
    # Two present slots both argmax to id 7 (slot0 more confident); a third wants id 3 (even more
    # confident than slot1). Greedy-by-confidence must: keep 7 on slot0, give slot2 its 3, and push
    # slot1 to some *other* id -> no duplicates among present slots, most-confident assignment wins.
    V = 10
    logits = torch.full((1, 4, V), -5.0)
    logits[0, 0, 7] = 10.0   # slot0: most confident, wants 7
    logits[0, 1, 7] = 5.0    # slot1: also wants 7 (less confident) -> must yield
    logits[0, 1, 3] = 4.0    # slot1: its second choice is 3 ...
    logits[0, 2, 3] = 8.0    # slot2: wants 3 and is MORE confident than slot1 -> slot2 keeps 3
    present = np.array([[True, True, True, False]])
    out = _dedup_slot_ids(logits, present)
    assert out.shape == (1, 4)
    assert out[0, 0] == 7                      # highest-confidence slot keeps its argmax
    assert out[0, 2] == 3                      # next-confident keeps its (distinct) argmax
    ids = out[0][present[0]].tolist()
    assert len(ids) == len(set(ids))           # no duplicate ids among present slots
    assert out[0, 1] not in (7, 3)             # the ambiguous slot got pushed to a free id


def test_relic_slot_dedup_noop_when_already_unique():
    V = 12
    logits = torch.full((1, 3, V), -5.0)
    for s, cid in enumerate((4, 9, 1)):
        logits[0, s, cid] = 9.0
    present = np.array([[True, True, True]])
    out = _dedup_slot_ids(logits, present)
    assert out[0].tolist() == [4, 9, 1]        # distinct argmaxes are left untouched


def test_relic_set_head_forward_shape_and_state_dict():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    model = _set_relic_model()
    assert model.cfg["relic_head"] == "set"
    _z, out = model(batch)
    # The relic type now emits ONE multi-hot vector over the catalog, no per-slot cat/presence.
    vocab = S.TYPE_BY_NAME["relic"].cat_cols[0][1]
    assert out["relic"]["set_logits"].shape == (2, vocab)
    assert "cat" not in out["relic"] and "presence" not in out["relic"]
    # No relic per-slot head tensors exist; a relic set_head weight does.
    keys = set(model.state_dict())
    assert any(k.startswith("decoder.heads.relic.set_head") for k in keys)
    assert not any(k.startswith("decoder.heads.relic.cat_heads") for k in keys)
    assert not any(k.startswith("decoder.heads.relic.presence_head") for k in keys)


def test_relic_set_head_topk_decode_no_duplicates_and_cardinality():
    model = _set_relic_model()
    _z, out = model(_batch([_state()]))
    vocab = out["relic"]["set_logits"].shape[-1]
    for chosen in ([5, 50, 120], [3, 7, 11, 40, 200]):
        logits = torch.full((1, vocab), -20.0)     # sigmoid ~0
        for rid in chosen:
            logits[0, rid] = 20.0                   # sigmoid ~1
        out["relic"]["set_logits"] = logits
        arrays = reconstruct_arrays(out)
        relics = tokens.detokenize(arrays[0])["relics"]
        # k = round(sum sigmoid) == len(chosen); top-k are exactly the chosen ids, sorted, no dups.
        assert sorted(relics) == sorted(chosen)
        assert len(relics) == len(set(relics))


def test_relic_set_head_decode_never_duplicates_on_random_logits():
    torch.manual_seed(0)
    model = _set_relic_model()
    _z, out = model(_batch([_state(), _state(n_hand=2, n_enemies=2)]))
    vocab = out["relic"]["set_logits"].shape[-1]
    out["relic"]["set_logits"] = torch.randn(2, vocab) * 3.0
    for arr in reconstruct_arrays(out):
        relics = tokens.detokenize(arr)["relics"]
        assert len(relics) == len(set(relics))
        assert len(relics) <= tokens.MAX_RELICS


def test_relic_set_head_loss_and_report_handoff():
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]),
                    _state(n_hand=3, n_enemies=2, relics=["BurningBlood"])])
    model = _set_relic_model()
    _z, out = model(batch)
    # compute_losses handles the set head (BCE folded into the categorical bucket) and stays finite.
    losses = M.compute_losses(batch, out)
    for k in ("loss", "loss_categorical", "loss_numeric", "loss_presence"):
        assert torch.isfinite(losses[k])
    # The full report card + detokenize hand-off still works with the set head.
    pairs = report.report_pairs(batch, out)
    assert set(pairs) == set(report.METRIC_NAMES)
    canon = tokens.detokenize(reconstruct_arrays(out)[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


def test_relic_set_head_loss_trains():
    torch.manual_seed(0)
    states = [_state(n_hand=i % 3 + 1, relics=["BurningBlood", "Anchor"][: i % 2 + 1]) for i in range(6)]
    batch = _batch(states)
    model = _set_relic_model()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = last = None
    for i in range(40):
        _z, out = model(batch)
        assert "set_logits" in out["relic"]
        losses = M.compute_losses(batch, out)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        if i == 0:
            first = float(losses["loss"])
        last = float(losses["loss"])
    assert last < first, f"set-head loss did not drop: {first:.3f} -> {last:.3f}"


def test_relic_head_default_slots_state_dict_identical():
    # Default (no relic_head arg) must be byte-identical to explicit relic_head="slots".
    torch.manual_seed(0)
    default = _small_model()
    torch.manual_seed(0)
    explicit = M.WorldModelAE(d_model=64, n_heads=2, enc_layers=2, dec_layers=2, n_pool_layers=1,
                              n_latents=4, z_dim=128, simnorm_group=8, cat_dim=16, n_mem=8,
                              relic_head="slots")
    assert default.cfg["relic_head"] == "slots"
    ka, kb = default.state_dict(), explicit.state_dict()
    assert ka.keys() == kb.keys()
    for k in ka:
        assert ka[k].shape == kb[k].shape and torch.equal(ka[k], kb[k])
    # Slots mode still emits per-slot relic cat + presence (never a set head).
    _z, out = default(_batch([_state()]))
    assert "set_logits" not in out["relic"]
    assert len(out["relic"]["cat"]) == 1 and "presence" in out["relic"]


def test_dedup_default_off_is_byte_identical_reconstruction():
    # reconstruct_arrays(dedup=False) is the default and unchanged from before the flag existed.
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]), _state(n_enemies=2)])
    model = _small_model()
    _z, out = model(batch)
    a = reconstruct_arrays(out)
    b = reconstruct_arrays(out, dedup=False)
    for da, db in zip(a, b):
        assert set(da) == set(db)
        for k in da:
            assert np.array_equal(da[k], db[k])


def test_checkpoint_relic_head_stamp_and_rejection(tmp_path):
    # Set-head checkpoint stamps relic_head=set (top-level + config) and rejects a slots load.
    model = _set_relic_model()
    path = str(tmp_path / "wm_set.pt")
    M.save_checkpoint(path, model, step=4)
    _loaded, meta = M.load_checkpoint(path, "cpu")
    assert meta["relic_head"] == "set"
    assert meta["config"]["relic_head"] == "set"
    try:
        M.load_checkpoint(path, "cpu", expect_relic_head="slots")
        assert False, "expected a relic_head-mismatch rejection"
    except ValueError as e:
        assert "relic_head" in str(e)
    # A slots (default) checkpoint stamps slots and rejects a set load.
    m2 = _small_model()
    p2 = str(tmp_path / "wm_slots.pt")
    M.save_checkpoint(p2, m2, step=1)
    _l2, meta2 = M.load_checkpoint(p2, "cpu")
    assert meta2["relic_head"] == "slots"
    try:
        M.load_checkpoint(p2, "cpu", expect_relic_head="set")
        assert False, "expected a relic_head-mismatch rejection"
    except ValueError as e:
        assert "relic_head" in str(e)
