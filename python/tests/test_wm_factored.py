"""Unit tests for the T3 factored expert autoencoder (:mod:`lts2_agent.wm.experts` +
:mod:`lts2_agent.wm.model_factored`). Synthetic tokenized states only — CPU, no C# host.

Covers: per-expert forward shapes; tier-1 scalar codec exact-by-construction (identity with RANDOM
weights, no training); range-bin numeric heads; relic set-head no-duplicates; card population-row
round-trip through reconstruct_arrays -> detokenize; slice-layout stamp + mismatch rejection; the
--arch mono/factored separation; and eval.expert_dist partitioning state_dist + eval.scalar_exact == 1.
"""

from __future__ import annotations

import numpy as np
import torch

from lts2_agent import tokens
from lts2_agent.wm import model as M
from lts2_agent.wm import model_factored as MF
from lts2_agent.wm import report
from lts2_agent.wm import spec as S
from lts2_agent.wm.decoder import reconstruct_arrays
from lts2_agent.wm.encoder import simnorm
from lts2_agent.wm.experts import EXPERT_ORDER, EXPERT_TYPES, ScalarCodec

# Reuse the synthetic state/batch builders from the monolith test module.
from tests.test_wm_encdec import _batch, _state


def _small() -> MF.FactoredWorldModelAE:
    # Narrow slices (divisible by simnorm_group=8) keep the CPU test light while exercising every expert.
    return MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, slice_widths={"creatures": 128, "cards": 256, "relics": 64, "potions": 32,
                                  "orbs": 32})


# ==================================================================================================
# Forward shapes + latent contract.
# ==================================================================================================

def test_forward_shapes_and_latent_dim():
    batch = _batch([_state(), _state(n_hand=3, n_enemies=2)])
    m = _small()
    z, out = m(batch)
    # Latent is the concatenation of all expert slices; layout offsets tile it exactly.
    assert z.shape == (2, m.latent_dim)
    assert m.latent_dim == m.scalars.width + 128 + 256 + 64 + 32 + 32
    off = 0
    for name in EXPERT_ORDER:
        a, b = m.slice_layout[name]
        assert a == off
        off = b
    assert off == m.latent_dim
    # Every token type is decoded with the monolith's per-type output structure.
    for t in S.TYPES:
        o = out[t.name]
        if t.name == "relic":
            assert o["set_logits"].shape == (2, t.cat_cols[0][1])
            continue
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
    for name in ("creatures", "cards", "relics", "potions", "orbs"):
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
    ex = m.experts["creatures"]
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
# Relic set head — duplicate-free by construction.
# ==================================================================================================

def test_relic_set_head_forward_and_no_duplicates():
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]), _state(n_enemies=2)])
    m = _small()
    _z, out = m(batch)
    vocab = S.TYPE_BY_NAME["relic"].cat_cols[0][1]
    assert out["relic"]["set_logits"].shape == (2, vocab)
    assert "cat" not in out["relic"] and "presence" not in out["relic"]
    # Random logits still decode a duplicate-free relic set within the cap.
    torch.manual_seed(0)
    out["relic"]["set_logits"] = torch.randn(2, vocab) * 3.0
    for arr in reconstruct_arrays(out):
        relics = tokens.detokenize(arr)["relics"]
        assert len(relics) == len(set(relics))
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
        assert set(losses) == {"loss", "loss_categorical", "loss_numeric", "loss_presence"}
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
    # A config mismatch (different relic head) is rejected.
    slots = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, relic_head="slots",
        slice_widths={"creatures": 128, "cards": 256, "relics": 64, "potions": 32, "orbs": 32})
    try:
        MF.init_expert_from(slots, "relics", path)
        assert False, "expected a config-mismatch rejection"
    except ValueError as e:
        assert "architecture differs" in str(e) or "config" in str(e)


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
    assert torch.equal(oc["relic"]["set_logits"], oa["relic"]["set_logits"])
    assert torch.equal(oc["card"]["cat"][0], ob["card"]["cat"][0])
    assert not torch.equal(oc["relic"]["set_logits"], ob["relic"]["set_logits"])


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
# Relic slots variant (inside the factored RelicExpert): forward + no duplicates via dedup.
# ==================================================================================================

def test_relic_slots_variant_forward_loss_and_dedup():
    m = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, relic_head="slots",
        slice_widths={"creatures": 128, "cards": 256, "relics": 64, "potions": 32, "orbs": 32})
    assert m.relic_head == "slots"
    batch = _batch([_state(relics=["BurningBlood", "Anchor"]), _state(n_enemies=2)])
    _z, out = m(batch)
    # Slot head emits per-slot categoricals + presence (NOT a set head).
    assert "set_logits" not in out["relic"]
    assert "cat" in out["relic"] and "presence" in out["relic"]
    # Loss is finite and includes the relic term.
    losses = MF.compute_losses(batch, out, m)
    assert all(torch.isfinite(v) for v in losses.values())
    # Decode with inference dedup: no relic id repeats within a state.
    torch.manual_seed(0)
    for arr in reconstruct_arrays(out, dedup=True):
        relics = tokens.detokenize(arr)["relics"]
        assert len(relics) == len(set(relics))


def test_relic_slots_variant_composes_with_default_experts(tmp_path):
    from lts2_agent.wm import compose as CMP
    slots = MF.FactoredWorldModelAE(
        d_model=64, n_heads=2, enc_layers=1, dec_layers=1, pool_layers=1, pool_latents=2, n_mem=4,
        cat_dim=16, relic_head="slots",
        slice_widths={"creatures": 128, "cards": 256, "relics": 64, "potions": 32, "orbs": 32})
    base = _small()   # relic_head="set"
    ps = str(tmp_path / "slots.pt"); pb = str(tmp_path / "base.pt")
    MF.save_checkpoint(ps, slots, step=1); MF.save_checkpoint(pb, base, step=1)
    out = str(tmp_path / "comp.pt")
    CMP.compose(out, {"relics": ps}, base=pb)
    comp, meta = MF.load_checkpoint(out)
    # The composite adopts the slots relic head; its non-relic experts stay the set-run defaults.
    assert comp.relic_head == "slots"
    _z, o = comp(_batch([_state(relics=["Anchor"])]))
    assert "set_logits" not in o["relic"]


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
