"""Unit tests for the T3 factored expert autoencoder (:mod:`lts2_agent.wm.experts` +
:mod:`lts2_agent.wm.model_factored`). Synthetic tokenized states only — CPU, no C# host.

Covers: per-expert forward shapes; tier-1 scalar codec exact-by-construction (identity with RANDOM
weights, no training); range-bin numeric heads; relic set-head no-duplicates; card population-row
round-trip through reconstruct_arrays -> detokenize; slice-layout stamp + mismatch rejection; the
--arch mono/factored separation; and eval.expert_dist partitioning state_dist + eval.scalar_exact == 1.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from lts2_agent import tokens
from lts2_agent.wm import cache as C
from lts2_agent.wm import data as D
from lts2_agent.wm import model as M
from lts2_agent.wm import model_factored as MF
from lts2_agent.wm import overfit as OF
from lts2_agent.wm import report
from lts2_agent.wm import spec as S
from lts2_agent.wm.decoder import reconstruct_arrays
from lts2_agent.wm.encoder import simnorm
from lts2_agent.wm.experts import (DIGIT_BASE, DIGIT_MIN_BINS, EXPERT_ORDER, EXPERT_TYPES,
                                   RangeBinHeads, ScalarCodec, t_symlog)

# Reuse the synthetic state/batch builders from the monolith test module.
from tests.test_wm_encdec import _batch, _state


# The creature family is three parameter-disjoint experts (owner ruling 2026-07-18): creature-stats,
# creature-powers, creature-intents. Narrow test slices for all three keep the CPU suite light.
_CREATURE_EXPERTS = ["creature-stats", "creature-powers", "creature-intents"]
_TEST_WIDTHS = {"creature-stats": 64, "creature-powers": 64, "creature-intents": 32,
                "cards": 256, "relics": 64, "potions": 32, "orbs": 32}


def _small() -> MF.FactoredWorldModelAE:
    # Narrow slices (divisible by simnorm_group=8) keep the CPU test light while exercising every expert.
    return MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS))


# ==================================================================================================
# Forward shapes + latent contract.
# ==================================================================================================

def test_forward_shapes_and_latent_dim():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    m = _small()
    z, out = m(batch)
    # Latent is the concatenation of all expert slices; layout offsets tile it exactly.
    assert z.shape == (2, m.latent_dim)
    # scalars + creature-stats/powers/intents + cards + relics + potions + orbs (EXPERT_ORDER).
    assert m.latent_dim == m.scalars.width + 64 + 64 + 32 + 256 + 64 + 32 + 32
    off = 0
    for name in EXPERT_ORDER:
        a, b = m.slice_layout[name]
        assert a == off
        off = b
    assert off == m.latent_dim
    # Every token type is decoded with the monolith's per-type output structure.
    for t in S.TYPES:
        o = out[t.name]
        assert len(o["cat"]) == len(t.cat_cols)
        for c, (_, vocab) in zip(o["cat"], t.cat_cols):
            assert c.shape[-1] == vocab
        if t.num_width:
            assert o["num"].shape[-1] == t.num_width
        if t.mask_key:
            assert o["presence"].shape == (2, t.max_slots)


def test_learned_slices_are_simnorm_and_scalars_is_binary_code():
    batch = _batch([_state()])
    m = _small()
    slices = m.encode_slices(batch)
    # Learned expert slices are SimNorm'd (grouped simplices).
    for name in (*_CREATURE_EXPERTS, "cards", "relics", "potions", "orbs"):
        g = slices[name].reshape(1, -1, m.cfg["simnorm_group"])
        assert torch.allclose(g.sum(-1), torch.ones_like(g.sum(-1)), atol=1e-5)
    # The scalar slice is a deterministic {0,1} code (no SimNorm — that would break exactness).
    sc = slices["scalars"]
    assert torch.all((sc == 0) | (sc == 1))


# ==================================================================================================
# Tier-1: exact by construction, with RANDOM weights and NO training.
# ==================================================================================================

def _random_scalar_batch(B: int, seed: int):
    """A synthetic batch of ONLY the global+pending arrays, filled with random IN-RANGE integers
    (symlog-encoded like the tokenizer), for the tier-1 exactness check."""
    rng = np.random.default_rng(seed)
    gi = np.zeros((B, 1, len(tokens.GLOBAL_IDX)), np.int32)
    for c, (_, vocab) in enumerate(S.TYPE_BY_NAME["global"].cat_cols):
        gi[:, 0, c] = rng.integers(0, vocab, size=B)
    gvals = np.zeros((B, len(tokens.GLOBAL_NUM)), np.int64)
    gn = np.zeros((B, 1, len(tokens.GLOBAL_NUM)), np.float32)
    for j, col in enumerate(tokens.GLOBAL_NUM):
        r = S.NUMERIC_RANGES["global"][col]
        v = rng.integers(r.lo, r.hi + 1, size=B)
        gvals[:, j] = v
        gn[:, 0, j] = [tokens.symlog(float(x)) for x in v]
    # pending: [present, minSelect, maxSelect, isUpgradeSelection]
    pv = np.zeros((B, 1, 4), np.float32)
    present = rng.integers(0, 2, size=B)
    mn = rng.integers(0, 11, size=B)
    mx = rng.integers(0, 101, size=B)
    up = rng.integers(0, 2, size=B)
    pv[:, 0, 0] = present
    pv[:, 0, 1] = [tokens.symlog(float(x)) for x in mn]
    pv[:, 0, 2] = [tokens.symlog(float(x)) for x in mx]
    pv[:, 0, 3] = up
    batch = {"global_idx": torch.tensor(gi), "global_num": torch.tensor(gn),
             "pending": torch.tensor(pv)}
    return batch, gi[:, 0], gvals, np.stack([present, mn, mx, up], axis=1)


def test_scalar_codec_exact_roundtrip_random_weights_no_training():
    B = 64
    batch, gi, gvals, pvals = _random_scalar_batch(B, seed=0)
    codec = ScalarCodec()   # parameter-free; there is nothing to train
    z = codec.encode(batch)
    out = codec.decode(z)
    # Global categoricals: argmax of the decoded (one-hot) logits recovers the exact enum index.
    for c in range(len(S.TYPE_BY_NAME["global"].cat_cols)):
        pred = out["global"]["cat"][c].argmax(-1)[:, 0].numpy()
        assert np.array_equal(pred, gi[:, c]), f"global cat col {c} not exact"
    # Global numerics: symexp+round of the decoded block recovers the exact integer.
    gnum = out["global"]["num"][:, 0, :].numpy()
    gdec = np.round(np.sign(gnum) * np.expm1(np.abs(gnum))).astype(np.int64)
    assert np.array_equal(gdec, gvals), "global numerics not exact by construction"
    # Pending: present flag, min, max, isUpgrade.
    pn = out["pending"]["num"][:, 0, :].numpy()
    assert np.array_equal((pn[:, 0] >= 0.5).astype(int), pvals[:, 0])
    assert np.array_equal(np.round(np.sign(pn[:, 1]) * np.expm1(np.abs(pn[:, 1]))).astype(int), pvals[:, 1])
    assert np.array_equal(np.round(np.sign(pn[:, 2]) * np.expm1(np.abs(pn[:, 2]))).astype(int), pvals[:, 2])
    assert np.array_equal(np.round(pn[:, 3]).astype(int), pvals[:, 3])


def test_scalar_exact_metric_is_one_with_random_factored_model():
    # Even a randomly-initialized full factored model reconstructs the tier-1 fields exactly (the codec
    # is independent of the learned experts) -> eval.scalar_exact pins to 1.0 from the first val pass.
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2), _state(n_enemies=2)])
    m = _small()
    _z, out = m(batch)
    pairs = report.report_pairs(batch, out, experts=True)
    overall = report.aggregate(pairs)
    assert overall["scalar_exact"] == 1.0
    # scalars is also the lowest-error expert (exact -> its expert_dist counts only cat/kw... here 0).
    assert overall["expert_dist::scalars"] == 0.0


# ==================================================================================================
# Range-bin numeric heads — exact decode when the head points at the right bin.
# ==================================================================================================

def test_range_bin_head_exact_when_argmax_hits_target_bin():
    batch = _batch([_state(damage=6) if False else _state(n_hand=2, n_enemies=2)])
    m = _small()
    ex = m.experts["creature-stats"]
    head = ex.heads["creature"]
    num = batch["creature_num"]                                   # [B, slots, W] symlog block
    tgt_bins = head.bin_targets(num)                             # exact target bins
    # Build one-hot logits at the target bin for every field, then decode: must reproduce the integers.
    W = len(head.num_cols)
    fake = {"num_bin_logits": []}
    for f in range(W):
        nb = int(head._nbins[f])
        oh = torch.zeros(num.shape[0], num.shape[1], nb)
        oh.scatter_(-1, tgt_bins[..., f:f + 1], 10.0)
        fake["num_bin_logits"].append(oh)
    idxs = torch.stack([lg.argmax(-1) for lg in fake["num_bin_logits"]], dim=-1)
    val = head._lo + idxs.float() * head._res
    dec = torch.where(head._is_raw, val, torch.sign(val) * torch.log1p(torch.abs(val)))
    # Integer round-trip of the decoded block matches the integer round-trip of the target block.
    di = np.round(np.sign(dec.numpy()) * np.expm1(np.abs(dec.numpy())))
    ti = np.round(np.sign(num.numpy()) * np.expm1(np.abs(num.numpy())))
    mask = batch["creature_mask"].numpy().astype(bool)
    assert np.array_equal(di[mask], ti[mask])


# ==================================================================================================
# Relic expert — positional per-slot decode (v5): duplicates legal, acquisition order carried.
# ==================================================================================================

def test_relic_expert_positional_slot_head():
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]), _state(n_enemies=2)])
    m = _small()
    _z, out = m(batch)
    t = S.TYPE_BY_NAME["relic"]
    # No set head: the relic expert emits per-slot categoricals (relicIndex + slot) + presence, exactly
    # like the other set experts.
    assert "set_logits" not in out["relic"]
    assert len(out["relic"]["cat"]) == len(t.cat_cols) == 2
    assert out["relic"]["cat"][1].shape[-1] == tokens.MAX_RELICS   # the positional `slot` head
    assert "presence" in out["relic"]
    # Reconstruct -> detokenize hands off cleanly (duplicates are legal, no dedup).
    for arr in reconstruct_arrays(out):
        relics = tokens.detokenize(arr)["relics"]
        assert len(relics) <= tokens.MAX_RELICS


# ==================================================================================================
# Card population rows round-trip through the reconstruct -> detokenize hand-off.
# ==================================================================================================

def test_card_rows_roundtrip_reconstruct_detokenize():
    from tests.test_wm_encdec import _card
    st = _state(n_hand=1)
    st["players"][0]["combatState"]["drawPile"] = [_card(damage=6, baseDamage=6) for _ in range(6)]
    batch = _batch([st])
    m = _small()
    _z, out = m(batch)
    # Full report card + detokenize both work with the factored outputs.
    pairs = report.report_pairs(batch, out, experts=True)
    assert set(report.METRIC_NAMES).issubset(set(pairs))
    canon = tokens.detokenize(reconstruct_arrays(out)[0])
    assert set(canon) == {"global", "pending", "cards", "creatures", "orbs", "relics", "potions"}


# ==================================================================================================
# Loss: finite, trains, and the three dashboard streams are present.
# ==================================================================================================

def test_loss_finite_and_overfits_one_batch():
    torch.manual_seed(0)
    states = [_state(n_hand=i % 4 + 1, n_enemies=i % 2 + 1) for i in range(6)]
    batch = _batch(states)
    m = _small()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = last = None
    for i in range(60):
        _z, out = m(batch)
        losses = MF.compute_losses(batch, out, m)
        assert set(losses) == {"loss", "loss_categorical", "loss_numeric", "loss_presence", "loss_zloss"}
        assert float(losses["loss_zloss"]) == 0.0                 # z-loss OFF by default
        assert all(torch.isfinite(v) for v in losses.values())
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        if i == 0:
            first = float(losses["loss"])
        last = float(losses["loss"])
    assert last < 0.6 * first, f"factored loss did not drop enough: {first:.3f} -> {last:.3f}"


# ==================================================================================================
# eval.expert_dist partitions state_dist exactly.
# ==================================================================================================

def test_expert_dist_partitions_state_dist():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2), _state(n_enemies=2)])
    m = _small()
    m.eval()
    _z, out = m(batch)
    pairs = report.report_pairs(batch, out, experts=True)
    overall = report.aggregate(pairs)
    # Every expert emits its own tagged pair; their partition sums (weighted) back to state_dist.
    tn = sum(pairs[f"expert_dist::{n}"][0].sum() for n in EXPERT_ORDER)
    td = sum(pairs[f"expert_dist::{n}"][1].sum() for n in EXPERT_ORDER)
    assert abs(float(tn / td) - overall["state_dist"]) < 1e-9
    # Token-type partition covers exactly the tokenizer's types (no double count / omission).
    covered = [tn for types in EXPERT_TYPES.values() for tn in types]
    assert sorted(covered) == sorted(tokens.TOKEN_TYPES)


# ==================================================================================================
# Checkpoint: arch + slice-layout stamp and mismatch rejection.
# ==================================================================================================

def test_checkpoint_roundtrip_and_slice_layout_stamp(tmp_path):
    m = _small()
    path = str(tmp_path / "wm_fac.pt")
    MF.save_checkpoint(path, m, step=9)
    loaded, meta = MF.load_checkpoint(path, "cpu")
    assert meta["arch"] == "factored"
    assert meta["step"] == 9
    assert meta["latent_dim"] == m.latent_dim
    names = [s["name"] for s in meta["slice_layout"]]
    assert names == EXPERT_ORDER
    # The reloaded model reproduces the latent shape.
    z, _ = loaded(_batch([_state()]))
    assert z.shape == (1, m.latent_dim)


def test_checkpoint_rejects_tokenizer_and_slice_mismatch(tmp_path):
    import json
    m = _small()
    path = str(tmp_path / "wm_fac.pt")
    MF.save_checkpoint(path, m, step=1)
    # Tokenizer-signature mismatch rejects loudly.
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    good_sig = meta["tokenizer_signature"]
    meta["tokenizer_signature"] = "tok-vDIFFERENT"
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    try:
        MF.load_checkpoint(path, "cpu")
        assert False, "expected a tokenizer-signature rejection"
    except ValueError as e:
        assert "different tokenizer" in str(e)
    # Slice-layout mismatch rejects loudly (the predictor addresses slices by these offsets).
    meta["tokenizer_signature"] = good_sig
    meta["slice_layout"][2]["width"] = 999999
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    try:
        MF.load_checkpoint(path, "cpu")
        assert False, "expected a slice-layout rejection"
    except ValueError as e:
        assert "slice layout" in str(e)


def test_factored_loader_rejects_mono_checkpoint(tmp_path):
    # A monolith checkpoint has no arch=factored stamp -> the factored loader rejects it (and vice versa
    # the mono loader would choke on the factored state_dict). This keeps the --arch paths separate.
    mono = M.WorldModelAE(d_model=64, n_heads=2, enc_layers=1, dec_layers=1, n_pool_layers=1,
                          n_latents=2, z_dim=64, simnorm_group=8, cat_dim=16, n_mem=4)
    path = str(tmp_path / "wm_mono.pt")
    M.save_checkpoint(path, mono, step=1)
    try:
        MF.load_checkpoint(path, "cpu")
        assert False, "expected an arch-mismatch rejection"
    except ValueError as e:
        assert "not 'factored'" in str(e)


def test_load_rejects_legacy_creatures_checkpoint(tmp_path):
    # Owner ruling (2026-07-18): the single 'creatures' expert was split into three parameter-disjoint
    # experts. An OLD checkpoint whose meta still names 'creatures' must fail LOUDLY and name the split,
    # never silently remap (there is no trained creatures expert worth preserving — it floored at 0.29).
    import json
    m = _small()
    path = str(tmp_path / "legacy.pt")
    MF.save_checkpoint(path, m, step=1)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    meta["experts"]["creatures"] = {"name": "creatures", "slice": [0, 768], "width": 768}   # pre-split
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    for loader in (MF.load_checkpoint, MF.read_meta):
        try:
            loader(path)
            assert False, "expected a legacy-expert rejection"
        except ValueError as e:
            msg = str(e)
            assert "legacy expert" in msg and "'creatures'" in msg
            assert all(s in msg for s in ("creature-stats", "creature-powers", "creature-intents"))


# ==================================================================================================
# Per-expert training: freeze/skip correctness.
# ==================================================================================================

def test_active_forward_skips_frozen_experts():
    # A subset forward decodes ONLY the active experts' token types and returns no full latent.
    batch = _batch([_state(n_hand=2, n_enemies=2), _state()])
    m = _small()
    m.eval()
    z, out = m(batch, active_experts=["relics"])
    assert z is None                                   # no concatenated latent for a partial pass
    assert set(out) == {"relic"}                       # cards/creatures/... skipped entirely
    # Full pass still decodes everything.
    z2, out2 = m(batch)
    assert z2.shape == (2, m.latent_dim)
    assert "card" in out2 and "creature" in out2


def test_frozen_experts_byte_identical_after_training_steps():
    # Train ONLY relics (optimizer owns just relic params, others frozen) — every frozen expert's
    # params must be byte-identical after several optimizer steps; the trained expert must move.
    torch.manual_seed(0)
    states = [_state(n_hand=i % 3 + 1, n_enemies=i % 2 + 1, relics=["BurningBlood", "Anchor"])
              for i in range(6)]
    batch = _batch(states)
    m = _small()
    active = ["relics"]
    # Mirror the trainer's freeze: requires_grad per active set; optimizer owns only trained params.
    trained_params = []
    for name, ex in m.experts.items():
        on = name in active
        for p in ex.parameters():
            p.requires_grad_(on)
        if on:
            trained_params += list(ex.parameters())
    frozen_before = {k: v.clone() for k, v in m.state_dict().items()
                     if not k.startswith("experts.relics.")}
    relic_before = {k: v.clone() for k, v in m.experts["relics"].state_dict().items()}
    opt = torch.optim.AdamW(trained_params, lr=1e-2)
    for _ in range(5):
        _z, out = m(batch, active_experts=active)
        losses = MF.compute_losses(batch, out, m, active=active)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
    after = m.state_dict()
    for k, v in frozen_before.items():
        assert torch.equal(v, after[k]), f"frozen param {k} changed under solo relic training"
    # The trained expert actually moved (at least one param differs).
    relic_after = m.experts["relics"].state_dict()
    assert any(not torch.equal(relic_before[k], relic_after[k]) for k in relic_before), \
        "trained relic expert did not change"


# ==================================================================================================
# Warm-start one expert from a full checkpoint's slice (--init-expert-from).
# ==================================================================================================

def test_init_expert_from_copies_slice(tmp_path):
    torch.manual_seed(1)
    src = _small()
    path = str(tmp_path / "src.pt")
    MF.save_checkpoint(path, src, step=42)
    torch.manual_seed(2)
    dst = _small()
    # Pre: dst's relic params differ from src's (different seeds).
    assert any(not torch.equal(a, b) for a, b in
               zip(src.experts["relics"].state_dict().values(),
                   dst.experts["relics"].state_dict().values()))
    stamp = MF.init_expert_from(dst, "relics", path)
    assert stamp["name"] == "relics"
    for k, v in src.experts["relics"].state_dict().items():
        assert torch.equal(v, dst.experts["relics"].state_dict()[k])
    # A config mismatch (different slice width) is rejected.
    narrow = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS, relics=32))
    try:
        MF.init_expert_from(narrow, "relics", path)
        assert False, "expected a slice-width-mismatch rejection"
    except ValueError as e:
        assert "width" in str(e) or "config" in str(e)


# ==================================================================================================
# Compose: assemble a full checkpoint from two runs' experts; eval matches per-expert sources.
# ==================================================================================================

def test_compose_roundtrip_matches_sources(tmp_path):
    from lts2_agent.wm import compose as CMP
    torch.manual_seed(3)
    a = _small(); a.eval()
    torch.manual_seed(4)
    b = _small(); b.eval()
    pa = str(tmp_path / "a.pt"); pb = str(tmp_path / "b.pt")
    MF.save_checkpoint(pa, a, step=10)
    MF.save_checkpoint(pb, b, step=20)
    out = str(tmp_path / "composite.pt")
    # relics from A, everything else from B.
    CMP.compose(out, {"relics": pa}, base=pb)
    comp, meta = MF.load_checkpoint(out)
    comp.eval()
    assert meta["composed_from"]["relics"] == pa
    batch = _batch([_state(n_hand=2, n_enemies=2, relics=["BurningBlood", "Anchor"]), _state()])
    with torch.no_grad():
        _za, oa = a(batch)
        _zb, ob = b(batch)
        _zc, oc = comp(batch)
    # Relic decode is byte-identical to source A; card decode byte-identical to source B.
    assert torch.equal(oc["relic"]["cat"][0], oa["relic"]["cat"][0])
    assert torch.equal(oc["card"]["cat"][0], ob["card"]["cat"][0])
    assert not torch.equal(oc["relic"]["cat"][0], ob["relic"]["cat"][0])


# ==================================================================================================
# eval.expert_exact — partition sanity + exact-by-construction scalars.
# ==================================================================================================

def test_expert_exact_matches_zero_dist_and_scalars_is_one():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2), _state(n_enemies=2)])
    m = _small(); m.eval()
    _z, out = m(batch)
    pairs = report.report_pairs(batch, out, experts=True)
    for ename in EXPERT_ORDER:
        exact = pairs[f"expert_exact::{ename}"][0]
        dist_num = pairs[f"expert_dist::{ename}"][0]
        # expert_exact[b] is exactly the "this expert had zero mismatched fields" indicator.
        assert np.array_equal(exact, (dist_num == 0.0).astype(np.float32))
    # scalars is exact by construction -> every state's scalar slice reconstructs exactly.
    assert report.aggregate(pairs)["expert_exact::scalars"] == 1.0


# ==================================================================================================
# --val-experts trained-only: focused per-expert report.
# ==================================================================================================

def test_trained_only_report_emits_focused_metrics():
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]), _state(n_enemies=2)])
    m = _small(); m.eval()
    _z, out = m(batch, active_experts=["relics"])   # only relics decoded
    pairs = report.report_pairs_experts_only(batch, out, ["relics"])
    assert "expert_dist::relics" in pairs
    assert "expert_exact::relics" in pairs
    assert "relic_set_f1" in pairs
    # No full-card metrics were computed (the whole point of trained-only).
    assert "card_id_top1" not in pairs and "expert_dist::cards" not in pairs
    ov = report.aggregate(pairs)
    assert 0.0 <= ov["expert_exact::relics"] <= 1.0
    assert 0.0 <= ov["relic_set_f1"] <= 1.0


# ==================================================================================================
# Two-hot (distance-aware) range-bin targets — solo-dynamics fix (M3.5).
# ==================================================================================================

def _stored_num_block(head, values_by_field):
    """Build a [1,1,W] stored numeric block from integer values (raw cols raw, others symlog), matching
    the tokenizer/head encoding so head.bin_targets recovers exactly these bins."""
    W = len(head.num_cols)
    block = np.zeros((1, 1, W), np.float32)
    for f, col in enumerate(head.num_cols):
        v = float(values_by_field[f])
        is_raw = bool(head._is_raw[f].item())
        block[0, 0, f] = v if is_raw else np.sign(v) * np.log1p(abs(v))
    return torch.tensor(block)


def test_twohot_targets_sum_to_one_and_expectation_recovers_value():
    m = _small()
    head = m.experts["creature-stats"].heads["creature"]
    W = len(head.num_cols)
    # Interior integer per field (>= half_width from both ends), so the symmetric kernel is untruncated.
    vals = []
    for f in range(W):
        lo = int(head._lo[f].item()); nb = int(head._nbins[f].item())
        vals.append(lo + min(nb - 1, max(0, nb // 3)))     # comfortably interior for the real ranges
    num = _stored_num_block(head, vals)
    centers = head.bin_targets(num)[0, 0]                   # [W] true bin index per field
    soft = head.soft_bin_targets(num)                      # list W of [1,1,nb_f]
    for f in range(W):
        dist = soft[f][0, 0]
        nb = int(head._nbins[f].item())
        # A proper distribution over this field's bins.
        assert dist.shape == (nb,)
        assert torch.allclose(dist.sum(), torch.tensor(1.0), atol=1e-6)
        # Expectation over bin centers recovers the true bin exactly (symmetric, interior).
        exp_bin = (dist * torch.arange(nb, dtype=dist.dtype)).sum()
        if nb > 2:                                          # flags (nb<=2) stay one-hot
            assert abs(float(exp_bin) - float(centers[f])) < 1e-5
            # Peak mass sits on the exact bin (argmax decode stays exact) with spread to neighbours.
            assert int(dist.argmax()) == int(centers[f])
            assert dist[int(centers[f])] < 1.0             # genuinely soft, not one-hot


def test_twohot_flag_fields_and_half_width_one_stay_one_hot():
    m = _small()
    head = m.experts["cards"].heads["card"]
    # Find a flag field (n_bins <= 2) — its soft target must remain one-hot.
    num = torch.zeros(1, 1, len(head.num_cols))
    soft = head.soft_bin_targets(num)
    centers = head.bin_targets(num)[0, 0]
    for f in range(len(head.num_cols)):
        if int(head._nbins[f].item()) <= 2:
            oh = soft[f][0, 0]
            assert float(oh[int(centers[f])]) == 1.0       # boolean never smeared
    # half_width=1 reproduces the hard one-hot for every field.
    soft1 = head.soft_bin_targets(num, half_width=1)
    for f in range(len(head.num_cols)):
        assert float(soft1[f][0, 0][int(centers[f])]) == 1.0


def test_twohot_and_hard_losses_both_finite_and_differ():
    torch.manual_seed(0)
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    m = _small()
    _z, out = m(batch)
    hard = MF.compute_losses(batch, out, m, num_targets="hard")
    two = MF.compute_losses(batch, out, m, num_targets="twohot")
    assert all(torch.isfinite(v) for v in hard.values())
    assert all(torch.isfinite(v) for v in two.values())
    # Same logits, different numeric target geometry -> different numeric loss (hard is legacy default).
    assert float(hard["loss_numeric"]) != float(two["loss_numeric"])


# ==================================================================================================
# Focus-present sampler — ratio + determinism (M3.5).
# ==================================================================================================

def _write_synthetic_shard(cache_dir, n, seed, empty_orbs_first):
    stacked = OF.synthetic_batch(n, seed=seed)
    stacked["orb_mask"] = stacked["orb_mask"].copy()
    stacked["orb_mask"][:empty_orbs_first] = False           # controlled present fraction for orbs
    feats = [{k: stacked[k][i] for k in M.BATCH_KEYS} for i in range(n)]
    split_dir = os.path.join(cache_dir, "train")
    os.makedirs(split_dir, exist_ok=True)
    C._write_shard(os.path.join(split_dir, "shard-00000.npz"), feats, [None] * n)


def test_expert_present_mask_unions_type_masks():
    stacked = OF.synthetic_batch(8, seed=0)
    stacked["orb_mask"] = stacked["orb_mask"].copy()
    stacked["orb_mask"][:4] = False
    pm = D.expert_present_mask(stacked, ["orbs"])
    assert pm.dtype == bool and pm.shape == (8,)
    assert not pm[:4].any() and pm[4:].all()


def test_focus_present_sampler_ratio_and_determinism(tmp_path):
    _write_synthetic_shard(str(tmp_path), n=400, seed=1, empty_orbs_first=200)
    B = 64
    n_present = round(0.9 * B)                                # 58 of 64
    g = D.focus_present_batches_cpu(str(tmp_path), "train", B, random.Random(0), ["orbs"], 0.9)
    b1, _ = next(g)
    b2, _ = next(g)
    for b in (b1, b2):
        assert int(D.expert_present_mask(b, ["orbs"]).sum()) == n_present   # exact target ratio
    # Same seed -> byte-identical stream (determinism).
    g2 = D.focus_present_batches_cpu(str(tmp_path), "train", B, random.Random(0), ["orbs"], 0.9)
    c1, _ = next(g2)
    assert np.array_equal(b1["orb_idx"], c1["orb_idx"])
    assert np.array_equal(b1["orb_mask"], c1["orb_mask"])


# ==================================================================================================
# Overfit-one-batch diagnostic — smoke on synthetic (the CLI wiring gate).
# ==================================================================================================

def test_overfit_batch_smoke_learns_on_synthetic():
    torch.manual_seed(0)
    m = _small()
    stacked = OF.synthetic_batch(16, seed=0)
    batch = M.to_tensors(stacked, torch.device("cpu"))
    res = OF.overfit_batch(m, batch, "orbs", steps=40, lr=3e-3, thresh=0.0,
                           num_targets="twohot", report_every=10, verbose=False)
    dists = [d for _, d, _ in res["history"]]
    assert all(np.isfinite(d) for d in dists)
    assert dists[-1] <= dists[0] + 1e-6                       # training reduces (never inflates) the dist
    # The numeric decode comparison runs and returns argmax/expectation fractions for the orb numerics.
    dec = OF.numeric_decode_compare(m, batch, "orbs")
    assert "orb" in dec and all(0.0 <= v <= 1.0 for v in dec["orb"])


def test_solo_expert_overfits_and_latent_not_collapsed():
    # Regression for the SimNorm saturation-runaway collapse (M3.5): the unbounded to_slice logits used to
    # run away in magnitude and saturate the grouped softmax to a state-INDEPENDENT one-hot (latent std ->
    # 0), so a solo expert could not overfit even a handful of distinct states. LayerNorm-before-SimNorm
    # fixes it — distinct states keep distinct latents and the reconstruction distance actually falls.
    torch.manual_seed(0)
    m = _small()
    st = OF.synthetic_batch(8, seed=2)
    st["orb_mask"] = np.zeros_like(st["orb_mask"])
    st["orb_mask"][:, :2] = True                     # 8 distinct 2-orb states
    batch = M.to_tensors(st, torch.device("cpu"))
    res = OF.overfit_batch(m, batch, "orbs", steps=500, lr=3e-3, thresh=0.0,
                           num_targets="hard", report_every=100, verbose=False)
    dists = [d for _, d, _ in res["history"]]
    assert dists[-1] < 0.5 * dists[0], f"solo overfit stalled (collapse?): {dists}"
    m.eval()
    with torch.no_grad():
        sl = m.experts["orbs"].encode(batch)
    assert float(sl.std(0).mean()) > 1e-3, "latent collapsed: distinct states share one slice"


def test_slice_norm_keeps_simplex_and_bounds_logits():
    # LayerNorm-before-SimNorm keeps the grouped-simplex contract AND bounds the pre-SimNorm input.
    m = _small()
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    slices = m.encode_slices(batch)
    for name in (*_CREATURE_EXPERTS, "cards", "relics", "potions", "orbs"):
        g = slices[name].reshape(slices[name].shape[0], -1, m.cfg["simnorm_group"])
        assert torch.allclose(g.sum(-1), torch.ones_like(g.sum(-1)), atol=1e-5)  # still simplices
    assert m.experts["orbs"].slice_norm.elementwise_affine is False              # no decay-shrinkable scale


# ==================================================================================================
# QK-norm expert trunks (Wortsman et al. 2023 fix (a)) + output z-loss (fix (b)).
#
# Motivation: the factored experts hit the textbook edge-of-stability transformer collapse at sustained
# flat LR — grad_norm creeps 1.5->1.9, one catastrophic spike, then a monotonic grad-norm ratchet
# (17->445->1171) into a degenerate all-absent presence solution. QK-norm bounds the attention logits;
# the z-loss bounds the output-logit growth. Backward compat is a hard requirement (wm.compose composes
# older, pre-fix experts), so an OLD checkpoint with no qk_norm key must rebuild the stock trunks and load
# byte-identically.
# ==================================================================================================

def _small_qk(qk_norm: bool) -> MF.FactoredWorldModelAE:
    return MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS), qk_norm=qk_norm)


def test_qk_norm_default_on_and_swaps_trunk_and_adds_params():
    from lts2_agent.wm.experts import (_QKNormAttention, _QKNormDecoder, _QKNormEncoder,
                                       _QKNormPoolLayer)
    on = _small_qk(True)
    off = _small_qk(False)
    assert on.cfg["qk_norm"] is True and off.cfg["qk_norm"] is False
    # Default construction is qk_norm ON for new runs.
    assert _small().cfg["qk_norm"] is True
    ex = on.experts["orbs"]
    assert isinstance(ex.enc_trunk, _QKNormEncoder)
    assert isinstance(ex.dec_trunk, _QKNormDecoder)
    assert isinstance(ex.pool[0], _QKNormPoolLayer)
    # A single attention block carries two QK LayerNorms over the head-dim (elementwise_affine=True).
    attn = ex.enc_trunk.layers[0].attn
    assert isinstance(attn, _QKNormAttention)
    assert attn.q_norm is not None and attn.q_norm.weight.shape == (attn.head_dim,)
    # qk_norm=False rebuilds the ORIGINAL stock trunk (no QK norm) — byte-identical arch to pre-fix runs.
    assert isinstance(off.experts["orbs"].enc_trunk, torch.nn.TransformerEncoder)
    # The only param delta is the QK LayerNorm weights+biases.
    delta = MF.param_count(on) - MF.param_count(off)
    assert delta > 0


def test_qk_norm_forward_backward_parity_smoke_trains_without_nan():
    # A qk_norm=True model trains a few steps on a tiny synth batch without NaN and the loss decreases.
    torch.manual_seed(0)
    states = [_state(n_hand=i % 4 + 1, n_enemies=i % 2 + 1) for i in range(6)]
    batch = _batch(states)
    m = _small_qk(True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = last = None
    for i in range(40):
        _z, out = m(batch)
        losses = MF.compute_losses(batch, out, m, num_targets="twohot")
        assert all(torch.isfinite(v) for v in losses.values())
        opt.zero_grad()
        losses["loss"].backward()
        # Every parameter must receive a finite gradient (the QK-norm block is differentiable end-to-end).
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
        if i == 0:
            first = float(losses["loss"])
        last = float(losses["loss"])
    assert last < first, f"qk-norm model did not train: {first:.3f} -> {last:.3f}"


def test_qk_norm_empty_category_forward_uses_cls_sentinel():
    # Empty orbs AND potions everywhere (no orb/potion tokens in any sample) must forward cleanly with
    # qk_norm=True: the CLS sentinel keeps >=1 valid attention key so the softmax never sees an all-padded
    # row (would be NaN). This is the exact edge case the sentinel mechanism exists for.
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    assert not batch["orb_mask"].any() and not batch["potion_mask"].any()   # truly empty categories
    m = _small_qk(True)
    m.eval()
    with torch.no_grad():
        z, out = m(batch)
    assert torch.isfinite(z).all()
    for o in out.values():
        for c in o.get("cat", []):
            assert torch.isfinite(c).all()
        if "presence" in o:
            assert torch.isfinite(o["presence"]).all()
    # The empty-category experts still produce a finite (non-NaN) latent slice.
    slices = m.encode_slices(batch)
    assert torch.isfinite(slices["orbs"]).all() and torch.isfinite(slices["potions"]).all()


def test_qk_norm_checkpoint_roundtrip_byte_identical(tmp_path):
    # Save a qk_norm=True model, reload via the same read-meta path compose/init use, and confirm the meta
    # records qk_norm and the reloaded state dict is byte-identical.
    torch.manual_seed(0)
    m = _small_qk(True)
    path = str(tmp_path / "qk.pt")
    MF.save_checkpoint(path, m, step=7)
    meta = MF.read_meta(path)
    assert meta["config"]["qk_norm"] is True
    assert meta["experts"]["orbs"]["config"]["qk_norm"] is True
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["qk_norm"] is True
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k


def test_old_checkpoint_without_qk_norm_key_loads_as_false(tmp_path):
    # Backward compat (hard requirement): a checkpoint predating the fix has NO qk_norm key in its meta.
    # Simulate by saving a qk_norm=False model and deleting the key from the saved meta, then reload — the
    # loader must construct qk_norm=False (stock trunks) and load byte-identically.
    import json
    torch.manual_seed(1)
    m = _small_qk(False)
    path = str(tmp_path / "old.pt")
    MF.save_checkpoint(path, m, step=3)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    del meta["config"]["qk_norm"]                                # pretend this checkpoint predates the fix
    for st in meta["experts"].values():
        st["config"].pop("qk_norm", None)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["qk_norm"] is False                        # defaulted for the pre-fix checkpoint
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k


def test_warm_start_new_qk_from_old_checkpoint_skips_trunk_copies_rest(tmp_path, capsys):
    # --init-expert-from an OLD (qk_norm=False) checkpoint into a NEW (qk_norm=True) model: the trunk param
    # names differ, so the trunk is skipped with a notice while the non-trunk weights still warm-start.
    import json
    torch.manual_seed(2)
    old = _small_qk(False)
    path = str(tmp_path / "old.pt")
    MF.save_checkpoint(path, old, step=5)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    del meta["config"]["qk_norm"]
    for st in meta["experts"].values():
        st["config"].pop("qk_norm", None)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    torch.manual_seed(3)
    new = _small_qk(True)
    before = {k: v.clone() for k, v in new.experts["relics"].state_dict().items()}
    MF.init_expert_from(new, "relics", path)
    out = capsys.readouterr().out
    assert "skipping" in out and "trunk" in out                  # printed notice for the skipped trunk
    after = new.experts["relics"].state_dict()
    # Non-trunk weights were copied from the old checkpoint; the renamed trunk params (QKV projections, QK
    # norms — absent from the stock trunk's state dict) kept the fresh init. (A few incidentally-named
    # trunk keys like out_proj share name+shape and copy harmlessly; we don't assert on those.)
    src = old.experts["relics"].state_dict()
    trunk = ("enc_trunk.", "pool.", "dec_trunk.")
    copied = skipped = 0
    for k in after:
        in_src = k in src and src[k].shape == after[k].shape
        if k.startswith(trunk) and not in_src:                   # genuinely renamed -> skipped
            skipped += 1
            assert torch.equal(after[k], before[k]), f"skipped trunk {k} should have kept its init"
        elif not k.startswith(trunk):
            assert in_src, f"non-trunk {k} unexpectedly absent from source"
            copied += 1
            assert torch.equal(after[k], src[k]), f"non-trunk {k} should have been copied"
    assert copied > 0 and skipped > 0


def test_z_loss_off_is_zero_and_on_penalizes_logits():
    # z-loss default OFF (0.0) leaves the loss dict byte-identical in value; turning it on adds a positive
    # penalty on the categorical + presence log-partitions, growing with the weight.
    torch.manual_seed(0)
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    m = _small()
    _z, out = m(batch)
    off = MF.compute_losses(batch, out, m, z_weight=0.0)
    assert float(off["loss_zloss"]) == 0.0
    base = float(off["loss"])
    on = MF.compute_losses(batch, out, m, z_weight=1e-3)
    assert float(on["loss_zloss"]) > 0.0
    assert abs(float(on["loss"]) - (base + float(on["loss_zloss"]))) < 1e-5   # added on top of the total
    stronger = MF.compute_losses(batch, out, m, z_weight=2e-3)
    assert float(stronger["loss_zloss"]) > float(on["loss_zloss"])            # scales with the weight
    # z-loss flows gradients (it is a real regularizer on the output logits).
    m.zero_grad()
    on["loss_zloss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in m.experts["cards"].parameters())


def test_mono_arch_unchanged_default_state_dict():
    # The monolith is untouched by the factored work: a default WorldModelAE built twice with the same
    # seed is byte-identical (the --arch mono path is the same code + weights as before).
    torch.manual_seed(0)
    a = M.WorldModelAE(d_model=64, n_heads=2, enc_layers=1, dec_layers=1, n_pool_layers=1,
                       n_latents=2, z_dim=64, simnorm_group=8, cat_dim=16, n_mem=4)
    torch.manual_seed(0)
    b = M.WorldModelAE(d_model=64, n_heads=2, enc_layers=1, dec_layers=1, n_pool_layers=1,
                       n_latents=2, z_dim=64, simnorm_group=8, cat_dim=16, n_mem=4)
    ka, kb = a.state_dict(), b.state_dict()
    assert ka.keys() == kb.keys()
    for k in ka:
        assert torch.equal(ka[k], kb[k])


# ==================================================================================================
# Cross-expert numeric fix (roadmap M3.5): eval decode mode, per-field loss norm, digit heads, n_mem.
# ==================================================================================================

def _perfect_num_logits(head: RangeBinHeads, centers: torch.Tensor) -> torch.Tensor:
    """Packed head logits that put all mass on the true bin/digits of ``centers`` ([..., W] long)."""
    import torch.nn.functional as F
    parts = []
    for f in range(len(head.num_cols)):
        nb = int(head._nbins[f].item())
        c = centers[..., f]
        dg = head._col_digit[f]
        if dg is not None:
            nd, base = dg
            dparts = []
            for d in range(nd):
                digit = torch.div(c, base ** d, rounding_mode="floor") % base
                dparts.append(F.one_hot(digit, base).float() * 30.0)
            parts.append(torch.cat(dparts, dim=-1))
        else:
            parts.append(F.one_hot(c, nb).float() * 30.0)
    return torch.cat(parts, dim=-1)


def test_digit_head_roundtrip_full_range_including_negatives():
    # power `amount` spans -30..250 (281 bins > DIGIT_MIN_BINS) -> a base-10, 3-digit column. Getting the
    # negative offset right is the crux: bin index = value - lo, decoded back must recover the exact
    # integer for EVERY value in the range, negatives included.
    head = RangeBinHeads(S.TYPE_BY_NAME["power"], 32, num_head="digits", num_decode="argmax")
    assert head._col_digit[0] is not None
    nd, base = head._col_digit[0]
    assert base == DIGIT_BASE and base ** nd >= 281 and base ** (nd - 1) < 281
    r = S.NUMERIC_RANGES["power"]["amount"]
    assert r.n_bins > DIGIT_MIN_BINS
    vals = torch.arange(r.lo, r.hi + 1)                       # full integer range incl negatives
    num = t_symlog(vals.float()).reshape(-1, 1, 1)           # symlog-stored (amount is not a raw col)
    centers = head.bin_targets(num)
    assert torch.equal(centers[..., 0].reshape(-1), (vals - r.lo))   # offset correct
    logits = _perfect_num_logits(head, centers)
    block = head.decode_num(logits)
    dec = (torch.sign(block) * torch.expm1(block.abs())).round().reshape(-1).long()
    assert torch.equal(dec, vals), "digit decode did not recover the exact integer across the full range"
    # Perfect digit logits -> ~0 per-digit CE (twohot degenerates to per-digit one-hot for a digit column).
    ce = head.numeric_field_ce(logits, num, num_targets="twohot", num_loss_norm="none")
    assert float(torch.stack(ce).mean()) < 1e-3


def test_decode_mode_argmax_vs_expected_on_synthetic_logits():
    # A diffuse two-hot-style distribution: argmax grabs the higher-mass neighbour, expected-bin decode
    # rounds the mass-weighted mean. A symmetric triangular distribution recovers the exact centre both ways.
    exp = RangeBinHeads(S.TYPE_BY_NAME["power"], 16, num_head="bins", num_decode="expected")
    arg = RangeBinHeads(S.TYPE_BY_NAME["power"], 16, num_head="bins", num_decode="argmax")
    nb = int(exp._nbins[0].item())
    lo = float(exp._lo[0].item())

    def to_bin(block):
        v = (torch.sign(block) * torch.expm1(block.abs())).round()
        return int((v[0, 0, 0] - lo))

    a = 40
    p = torch.zeros(1, 1, nb); p[0, 0, a] = 0.4; p[0, 0, a + 2] = 0.6    # mean a+1.2, mode a+2
    logits = (p + 1e-9).log()
    assert to_bin(arg.decode_num(logits)) == a + 2                       # argmax = highest-mass bin
    assert to_bin(exp.decode_num(logits)) == a + 1                       # expected = round(mean)
    p2 = torch.zeros(1, 1, nb); p2[0, 0, a - 1] = 0.25; p2[0, 0, a] = 0.5; p2[0, 0, a + 1] = 0.25
    l2 = (p2 + 1e-9).log()
    assert to_bin(arg.decode_num(l2)) == a and to_bin(exp.decode_num(l2)) == a   # symmetric: both exact


def test_num_loss_norm_logbins_divides_each_field_by_log_nbins():
    import math
    head = RangeBinHeads(S.TYPE_BY_NAME["card"], 24, num_head="bins")
    B, slots = 2, 3
    num = torch.zeros(B, slots, len(head.num_cols))          # valid in-range targets (bin = -lo)
    torch.manual_seed(0)
    logits = torch.randn(B, slots, int(sum(head._col_widths)))
    none = head.numeric_field_ce(logits, num, num_targets="hard", num_loss_norm="none")
    logb = head.numeric_field_ce(logits, num, num_targets="hard", num_loss_norm="logbins")
    for f in range(len(head.num_cols)):
        nb = int(head._nbins[f].item())
        assert torch.allclose(logb[f], none[f] / math.log(max(nb, 2)), atol=1e-6), f


def test_compute_losses_num_loss_norm_changes_numeric_term():
    torch.manual_seed(0)
    m = _small()
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    _z, out = m(batch)
    none = MF.compute_losses(batch, out, m, num_loss_norm="none")
    logb = MF.compute_losses(batch, out, m, num_loss_norm="logbins")
    assert all(torch.isfinite(v) for v in logb.values())
    assert float(none["loss_numeric"]) != float(logb["loss_numeric"])   # same logits, rescaled fields


def test_expert_overrides_n_mem_plumbs_and_roundtrips(tmp_path):
    # --expert-n-mem lands as expert_overrides[name]["n_mem"], sizing that expert's decoder memory + the
    # from_slice projection, and round-trips through the per-expert stamp (like dec_layers).
    m = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS), expert_overrides={"orbs": {"n_mem": 9}})
    assert m.experts["orbs"].n_mem == 9 and m.experts["cards"].n_mem == 4
    assert m.experts["orbs"].from_slice.out_features == 9 * 64
    path = str(tmp_path / "nmem.pt")
    MF.save_checkpoint(path, m, step=1)
    meta = MF.read_meta(path)
    assert meta["experts"]["orbs"]["config"]["n_mem"] == 9
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.experts["orbs"].n_mem == 9 and loaded.experts["cards"].n_mem == 4


def test_old_checkpoint_without_num_head_keys_loads_as_bins_argmax(tmp_path):
    # Backward compat (hard requirement): a checkpoint predating the numeric fix has neither num_head nor
    # num_decode in its meta. The loader must rebuild the flat "bins" head + "argmax" decode and load the
    # old state_dict byte-identically.
    import json
    torch.manual_seed(1)
    m = _small()                                            # defaults: bins head, expected decode
    path = str(tmp_path / "old.pt")
    MF.save_checkpoint(path, m, step=2)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    del meta["config"]["num_head"]; del meta["config"]["num_decode"]
    for st in meta["experts"].values():
        st["config"].pop("num_head", None); st["config"].pop("num_decode", None)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["num_head"] == "bins" and loaded.cfg["num_decode"] == "argmax"
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k


def test_digit_head_model_roundtrip_and_meta(tmp_path):
    m = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS), num_head="digits")
    ch = m.experts["creature-stats"].heads["creature"]
    assert any(d is not None for d in ch._col_digit)          # currentHp/maxHp (1000+ bins) -> digit heads
    path = str(tmp_path / "dig.pt")
    MF.save_checkpoint(path, m, step=3)
    meta = MF.read_meta(path)
    assert meta["config"]["num_head"] == "digits"
    assert meta["experts"]["creature-stats"]["config"]["num_head"] == "digits"
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["num_head"] == "digits"
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k
    # forward: a digit-carrying head decodes `num` but exposes no flat per-field `num_bin_logits`.
    batch = _batch([_state(n_enemies=2)])
    with torch.no_grad():
        _z, out = loaded(batch)
    assert "num" in out["creature"] and "num_bin_logits" not in out["creature"]
    assert "num_logits" in out["creature"]


def test_digit_head_compute_losses_finite_both_targets():
    torch.manual_seed(0)
    m = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=dict(_TEST_WIDTHS), num_head="digits")
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    _z, out = m(batch)
    for nt in ("hard", "twohot"):
        for norm in ("none", "logbins"):
            losses = MF.compute_losses(batch, out, m, num_targets=nt, num_loss_norm=norm)
            assert all(torch.isfinite(v) for v in losses.values()), (nt, norm)


# ==================================================================================================
# Numeric INPUT featurization (roadmap M3.5 INPUT half): --num-input {symlog,digits,fourier,both}.
#
# WHY: numerics enter the encoder as ONE symlog float per column, so symlog(500) vs symlog(501) differ by
# ~0.002 — large-value precision is lost at the INPUT (the decoder can't emit resolution the latent never
# received). digits/fourier ADD a high-resolution featurization derived exactly from the stored symlog
# float (no tokenizer/cache change). 'symlog' (default) is byte-identical to the pre-featurization model.
# ==================================================================================================

def _num_input_widths():
    return dict(_TEST_WIDTHS)


def test_num_input_featurization_roundtrip_including_negatives():
    # power `amount` spans -30..250 (281 bins) — the negative-offset case. Exact integer recovery from the
    # stored symlog float is the contract; digit + fourier features derive deterministically from it.
    from lts2_agent.wm.encoder import _TypeEmbedder, _DIGIT_BASE, _DIGIT_EMB_DIM
    st = MF._static_tables()
    emb = _TypeEmbedder(S.TYPE_BY_NAME["power"], 32, 16, st, num_input="both")
    assert len(emb._feat_meta) == 1
    m0 = emb._feat_meta[0]
    assert m0["col"] == "amount" and m0["lo"] == -30 and m0["nbins"] == 281 and m0["nd"] == 3
    r = S.NUMERIC_RANGES["power"]["amount"]
    vals = torch.arange(r.lo, r.hi + 1)                          # -30..250, negatives included
    num = t_symlog(vals.float()).reshape(-1, 1, 1)               # symlog-stored (amount is not a raw col)
    bins = emb._bins(num)[0]                                     # [N,1] recovered bin index
    assert torch.equal(bins.reshape(-1), (vals - r.lo).long())   # exact recovery incl the negative offset
    # Digit features: nd*_DIGIT_EMB_DIM wide, and exactly the per-digit embedding lookups of the recovered
    # bins (base-10 digits of the value-lo offset — the output-head convention).
    df = emb.digit_features(num)
    assert df.shape[-1] == m0["nd"] * _DIGIT_EMB_DIM
    parts = [emb.digit_embs[0][d](torch.div(bins, _DIGIT_BASE ** d, rounding_mode="floor") % _DIGIT_BASE)
             for d in range(m0["nd"])]
    assert torch.equal(df, torch.cat(parts, dim=-1))
    # Fourier features: 2*K wide, exactly sin/cos of the offset at the stored geometric frequencies.
    ff = emb.fourier_features(num)
    freqs = emb._fourier_freq_0
    assert ff.shape[-1] == 2 * freqs.numel()
    ang = (vals - r.lo).float().reshape(-1, 1, 1) * freqs
    assert torch.allclose(ff, torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1), atol=1e-6)


def test_num_input_symlog_is_byte_identical_default():
    # 'symlog' (the default) adds NO params and leaves the embedder proj shape unchanged — a factored model
    # built with num_input='symlog' is byte-identical to one built without the flag.
    from lts2_agent.wm.encoder import _TypeEmbedder
    st = MF._static_tables()
    plain = _TypeEmbedder(S.TYPE_BY_NAME["creature"], 32, 16, st)
    sym = _TypeEmbedder(S.TYPE_BY_NAME["creature"], 32, 16, st, num_input="symlog")
    assert plain.proj.in_features == sym.proj.in_features
    assert not sym._feat_meta and not hasattr(sym, "digit_embs") and not hasattr(sym, "fourier_proj")


def test_num_input_modes_forward_backward_smoke():
    # Each mode forwards + backwards on a tiny CPU batch with finite loss and finite grads; the non-symlog
    # modes add real (differentiable) params over symlog.
    torch.manual_seed(0)
    batch = _batch([_state(n_hand=2, n_enemies=2), _state(n_enemies=1)])
    counts = {}
    for mode in ("symlog", "digits", "fourier", "both"):
        m = MF.FactoredWorldModelAE(
            d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
            cat_dim=16, slice_widths=_num_input_widths(), num_input=mode)
        assert m.cfg["num_input"] == mode
        _z, out = m(batch)
        losses = MF.compute_losses(batch, out, m, num_targets="twohot")
        assert all(torch.isfinite(v) for v in losses.values()), mode
        m.zero_grad()
        losses["loss"].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()), mode
        counts[mode] = MF.param_count(m)
    assert counts["digits"] > counts["symlog"]
    assert counts["fourier"] > counts["symlog"]
    assert counts["both"] > counts["digits"] and counts["both"] > counts["fourier"]


def test_num_input_model_roundtrip_and_meta(tmp_path):
    # Meta stamps num_input model-wide + per expert; the reload is byte-identical.
    m = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=_num_input_widths(), num_input="both")
    ce = m.experts["creature-stats"].embedders["creature"]
    assert hasattr(ce, "digit_embs") and hasattr(ce, "fourier_proj")   # ranged expert got both
    # relics carry no numeric block -> no featurization params even in 'both'.
    re = m.experts["relics"].embedders["relic"]
    assert not re._feat_meta and not hasattr(re, "digit_embs") and not hasattr(re, "fourier_proj")
    path = str(tmp_path / "ni.pt")
    MF.save_checkpoint(path, m, step=3)
    meta = MF.read_meta(path)
    assert meta["config"]["num_input"] == "both"
    assert meta["experts"]["creature-stats"]["config"]["num_input"] == "both"
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["num_input"] == "both"
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k


def test_old_checkpoint_without_num_input_key_loads_as_symlog(tmp_path):
    # Backward compat (hard requirement): a checkpoint predating this fix has no num_input key. The loader
    # must default it to 'symlog' and load the old state_dict byte-identically.
    import json
    torch.manual_seed(1)
    m = _small()                                                 # default num_input='symlog'
    path = str(tmp_path / "old.pt")
    MF.save_checkpoint(path, m, step=2)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    del meta["config"]["num_input"]
    for st in meta["experts"].values():
        st["config"].pop("num_input", None)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)
    loaded, _ = MF.load_checkpoint(path, "cpu")
    assert loaded.cfg["num_input"] == "symlog"
    src, dst = m.state_dict(), loaded.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.equal(src[k], dst[k]), k


def test_init_expert_from_tolerates_num_input_diff_skips_embedder(tmp_path, capsys):
    # --init-expert-from across a num_input change: the per-type embedder shapes differ (extra digit params
    # + wider proj), so those are SKIPPED with a notice while the shared non-embedder weights warm-start.
    torch.manual_seed(2)
    src = _small()                                               # symlog embedders
    path = str(tmp_path / "src.pt")
    MF.save_checkpoint(path, src, step=5)
    torch.manual_seed(3)
    dst = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths=_num_input_widths(), num_input="digits")
    before = {k: v.clone() for k, v in dst.experts["creature-stats"].state_dict().items()}
    MF.init_expert_from(dst, "creature-stats", path)
    out = capsys.readouterr().out
    assert "skipping" in out and "embedder" in out
    after = dst.experts["creature-stats"].state_dict()
    src_sub = src.experts["creature-stats"].state_dict()
    copied = skipped = 0
    for k in after:
        in_src = k in src_sub and src_sub[k].shape == after[k].shape
        if k.startswith("embedders.") and not in_src:            # digit params / widened proj -> skipped
            skipped += 1
            assert torch.equal(after[k], before[k]), f"skipped embedder {k} should keep its init"
        elif not k.startswith(("embedders.", "enc_trunk.", "pool.", "dec_trunk.")):
            assert in_src, f"non-embedder {k} unexpectedly absent from source"
            copied += 1
            assert torch.equal(after[k], src_sub[k]), f"non-embedder {k} should have been copied"
    assert copied > 0 and skipped > 0


# ==================================================================================================
# DIAGNOSTIC --cards-static-only (STATIC-ONLY probe, roadmap wm-t3-factored): mask the TRANSIENT card
# numeric columns (all CARD_NUM except `upgraded`) out of BOTH the card training loss and the report's
# per-column mismatch. cardIndex/type/rarity/targetType/enchant/afflict/zone/slot/upgraded + keywords stay.
# ==================================================================================================

def _card_present_batch():
    """A batch with several PRESENT card rows (drawPile) so the transient-column mask has real slots to
    act on. Built via the shared _state/_card helpers (no reachability artifact needed)."""
    from tests.test_wm_encdec import _card
    st = _state(n_hand=2)
    st["players"][0]["combatState"]["drawPile"] = [_card(damage=6, baseDamage=6) for _ in range(5)]
    return _batch([st, _state(n_hand=1)])


def test_cards_transient_set_is_all_card_num_except_upgraded():
    from lts2_agent.wm.experts import CARD_STATIC_NUM_KEEP, CARD_TRANSIENT_NUM
    assert set(CARD_TRANSIENT_NUM) == set(tokens.CARD_NUM) - {"upgraded"}
    assert "upgraded" not in CARD_TRANSIENT_NUM
    assert tuple(tokens.CARD_NUM[j] for j in CARD_STATIC_NUM_KEEP) == ("upgraded",)


def test_cards_static_only_masks_transient_numeric_targets_in_loss():
    torch.manual_seed(0)
    m = _small()
    batch = _card_present_batch()
    assert batch["card_mask"].any()
    _z, out = m(batch, active_experts=["cards"])

    def numeric(b):
        return float(MF.compute_losses(b, out, m, active=["cards"],
                                       cards_static_only=True)["loss_numeric"])

    def total(b):
        return float(MF.compute_losses(b, out, m, active=["cards"],
                                       cards_static_only=True)["loss"])

    base_num, base_tot = numeric(batch), total(batch)
    # Perturbing a TRANSIENT column's TARGET (damage) — masked out, so the numeric (and total) loss is
    # byte-identical (out is fixed; only the target moved, and its CE is dropped from the mean).
    dmg = tokens.CARD_NUM.index("damage")
    bt = {k: v.clone() for k, v in batch.items()}
    bt["card_num"][..., dmg] += 2.0
    assert numeric(bt) == base_num
    assert total(bt) == base_tot
    # Perturbing the KEPT column (`upgraded`) DOES change the numeric loss (it survives the mask).
    up = tokens.CARD_NUM.index("upgraded")
    bu = {k: v.clone() for k, v in batch.items()}
    bu["card_num"][..., up] = 1.0 - bu["card_num"][..., up]
    assert numeric(bu) != base_num
    # Perturbing the cardIndex categorical TARGET changes the (categorical -> total) loss too.
    bc = {k: v.clone() for k, v in batch.items()}
    vocab = S.TYPE_BY_NAME["card"].cat_cols[0][1]
    bc["card_idx"][..., 0] = (bc["card_idx"][..., 0] + 1) % vocab
    assert total(bc) != base_tot


def _card_arrays(n_present: int = 3):
    """A per-sample array dict (the shape report._state_dist / detokenize consume) with ``n_present``
    left-packed card rows and everything else zeroed — used to drive the report's per-column mismatch
    with controlled PRESENT-on-both slots (an untrained model's presence rarely agrees, so a live forward
    can't exercise the 'both present' numeric-column comparison the mask acts on)."""
    return {
        "card_idx": np.zeros((tokens.MAX_CARDS, len(tokens.CARD_IDX)), np.int32),
        "card_num": np.zeros((tokens.MAX_CARDS, len(tokens.CARD_NUM)), np.float32),
        "card_kw": np.zeros((tokens.MAX_CARDS, len(tokens.KEYWORDS)), np.float32),
        "card_mask": np.array([i < n_present for i in range(tokens.MAX_CARDS)], bool),
    }


def test_cards_static_only_state_dist_ignores_transient_columns():
    # _state_dist is the report's per-column mismatch. With the mask on, a TRANSIENT numeric mismatch
    # (damage) is not counted and drops from the denominator; the KEPT `upgraded` mismatch still counts.
    from lts2_agent.wm.report import _state_dist
    types = [S.TYPE_BY_NAME["card"]]
    tgt = _card_arrays(3)
    # Identical arrays -> zero mismatch, positive denominator (present-both card fields).
    n0, d0 = _state_dist(_card_arrays(3), tgt, types=types, cards_static_only=True)
    assert n0 == 0.0 and d0 > 0.0
    # Perturb a TRANSIENT numeric (damage) in the prediction: masked distance stays 0; unmasked counts 3.
    dmg = tokens.CARD_NUM.index("damage")
    pred_t = _card_arrays(3); pred_t["card_num"][:3, dmg] = tokens.symlog(9)
    nt, dt = _state_dist(pred_t, tgt, types=types, cards_static_only=True)
    nu, du = _state_dist(pred_t, tgt, types=types, cards_static_only=False)
    assert nt == 0.0                                # transient column masked out of the numerator
    assert nu == 3.0                                # unmasked: 3 present-both slots differ on damage
    assert dt < du                                  # masked denominator drops the transient columns
    # Perturb the KEPT `upgraded` numeric: it survives the mask, so the masked distance counts it (3 slots).
    up = tokens.CARD_NUM.index("upgraded")
    pred_u = _card_arrays(3); pred_u["card_num"][:3, up] = 1.0
    nk, _dk = _state_dist(pred_u, tgt, types=types, cards_static_only=True)
    assert nk == 3.0


def test_cards_static_only_report_reduces_card_denominator():
    # End-to-end through report_pairs_experts_only: the masked run's card expert_dist denominator counts
    # fewer fields than the unmasked run's (the transient numerics are dropped from the field universe).
    torch.manual_seed(0)
    m = _small(); m.eval()
    batch = _card_present_batch()
    _z, out = m(batch, active_experts=["cards"])
    masked = report.report_pairs_experts_only(batch, out, ["cards"], cards_static_only=True)
    unmasked = report.report_pairs_experts_only(batch, out, ["cards"])
    assert masked["expert_dist::cards"][1].sum() < unmasked["expert_dist::cards"][1].sum()
    # v7: the granted-rows keyword slice is emitted for a cards-active run.
    assert "card_kw_granted_exact" in unmasked


def test_card_kw_granted_exact_metric_on_synthetic_case():
    # The granted-rows slice restricts to card rows whose ABSOLUTE keyword flags differ from their card's
    # PRINTED flags (rows carrying a runtime grant), and scores an exact all-7-bit keyword match. cardIndex
    # 0 has empty printed flags, so a row with any set target flag is 'granted' regardless of the catalog.
    K = len(tokens.KEYWORDS)
    slots = tokens.MAX_CARDS
    retain = tokens.KEYWORDS.index("Retain")
    exhaust = tokens.KEYWORDS.index("Exhaust")
    card_mask = np.zeros((1, slots), bool); card_mask[0, :3] = True
    tkw = np.zeros((1, slots, K), np.float32)
    tkw[0, 0, retain] = 1.0        # row0: granted (printed empty, absolute {Retain})
    # row1: empty -> absolute == printed empty -> NOT granted
    tkw[0, 2, exhaust] = 1.0       # row2: granted ({Exhaust})
    batch = {
        "card_idx": torch.zeros((1, slots, len(tokens.CARD_IDX)), dtype=torch.long),  # cardIndex 0
        "card_mask": torch.from_numpy(card_mask),
        "card_kw": torch.from_numpy(tkw),
    }
    plogits = np.full((1, slots, K), -10.0, np.float32)
    plogits[0, 0, retain] = 10.0   # row0 predicted exactly {Retain} -> exact
    # row2 predicted nothing -> misses Exhaust (wrong)
    outputs = {"card": {"kw": torch.from_numpy(plogits)}}
    num, den = report._card_kw_granted_pairs(batch, outputs)
    assert den[0] == 2.0           # rows 0 and 2 carry grants; row1 (no grant) excluded
    assert num[0] == 1.0           # only row0 reconstructs all 7 keyword bits exactly
    # A non-card run (no card kw head) yields no slice.
    assert report._card_kw_granted_pairs(batch, {"relic": {}}) is None


# ==================================================================================================
# DIAGNOSTIC flags plumb through the trainer CLI and stamp into the run config (MetricsWriter records
# vars(args) -> manifest.config), and are mutually composable.
# ==================================================================================================

def test_kw_pos_weight_flag_parses_and_scales_kw_bce():
    from lts2_agent import train_encdec as TE
    ap = TE.build_parser()
    assert ap.parse_args([]).kw_pos_weight == 1.0                       # default OFF
    assert ap.parse_args(["--kw-pos-weight", "5"]).kw_pos_weight == 5.0
    # Effect: with positive keyword targets present, weighting the positive class raises the kw BCE (and
    # hence the categorical loss) vs the default pos_weight 1.0.
    torch.manual_seed(0)
    m = _small(); m.eval()
    batch = {k: v.clone() for k, v in _card_present_batch().items()}
    batch["card_kw"][:] = 0.0
    batch["card_kw"][..., 0] = 1.0                                       # present cards "want" column 0 set
    _z, out = m(batch, active_experts=["cards"])
    base = float(MF.compute_losses(batch, out, m, active=["cards"])["loss_categorical"])
    up = float(MF.compute_losses(batch, out, m, active=["cards"], kw_pos_weight=5.0)["loss_categorical"])
    assert up > base


def test_diagnostic_flags_parse_and_are_composable():
    from lts2_agent import train_encdec as TE
    ap = TE.build_parser()
    args = ap.parse_args(["--cards-max-rows", "1", "--cards-static-only"])   # both together (composable)
    assert args.cards_max_rows == 1 and args.cards_static_only is True
    d = ap.parse_args([])                                                    # defaults: both OFF
    assert d.cards_max_rows is None and d.cards_static_only is False
    assert "cards_max_rows" in vars(args) and "cards_static_only" in vars(args)


def test_diagnostic_flags_stamp_into_metrics_config(tmp_path):
    import json
    from lts2_agent import train_encdec as TE
    from lts2_agent.metrics import MetricsWriter
    args = TE.build_parser().parse_args(["--cards-max-rows", "1", "--cards-static-only"])
    mw = MetricsWriter(run_dir=str(tmp_path), label="probe", argv=["train_encdec"],
                       config=vars(args), enabled=True)
    mw.close()
    with open(os.path.join(mw.run_dir, "manifest.json")) as f:
        cfg = json.load(f)["config"]
    assert cfg["cards_max_rows"] == 1 and cfg["cards_static_only"] is True
