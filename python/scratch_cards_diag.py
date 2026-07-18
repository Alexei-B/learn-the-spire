"""Scratch diagnostic (no product changes): where does the cards-expert error live?

Loads cards_cos50k.pt (factored, --train-experts cards, slice-width cards=2048, tok-v5, gitSha 44788ba
pre-creature-split). The current HEAD load_checkpoint would REJECT this checkpoint (legacy single
'creatures' expert). Verified out-of-band that tokens/catalog/spec/report/data/decoder/model are
byte-identical 44788ba..HEAD and the cards SetExpert/RangeBinHeads params + the _fill_cards generator are
unchanged, so we reconstruct just the cards expert at HEAD and load experts.cards.* faithfully.
CPU only.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                       # so "data/corpus" and the reachable table resolve like the trainer
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lts2_agent import tokens
from lts2_agent.wm import spec as S, experts as E, model_factored as MF, report, data as D, synth as SY
from lts2_agent.wm import model as M

CKPT = "C:/Users/Alexe/.claude/jobs/06eda07b/tmp/cards_cos50k.pt"
VAL_STATES = 2000
VAL_BATCH = 256

# ---- construct the cards expert exactly as FactoredWorldModelAE does, load its slice ----
meta = json.load(open(CKPT + ".meta.json"))
cfg = meta["config"]
print(f"ckpt step={meta['step']} tok={meta['tokenizer_signature'][:24]}... "
      f"cards_width={cfg['slice_widths']['cards']} qk_norm={cfg['qk_norm']}")
static_tables = MF._static_tables()
common = dict(d_model=cfg["d_model"], static_tables=static_tables, cat_dim=cfg["cat_dim"],
              n_heads=cfg["n_heads"], pool_layers=cfg["pool_layers"], pool_latents=cfg["pool_latents"],
              n_mem=cfg["n_mem"], simnorm_group=cfg["simnorm_group"], qk_norm=cfg["qk_norm"])
deep = dict(common, enc_layers=cfg["enc_layers"], dec_layers=cfg["dec_layers"])
cards = E.SetExpert("cards", ["card"], cfg["slice_widths"]["cards"], **deep)
sd = torch.load(CKPT, map_location="cpu")
sub = {k[len("experts.cards."):]: v for k, v in sd.items() if k.startswith("experts.cards.")}
missing, unexpected = cards.load_state_dict(sub, strict=False)
# strict=False tolerates non-persistent buffers; report any *parameter* gaps loudly
pnames = {n for n, _ in cards.named_parameters()}
bad = [k for k in missing if k in pnames] + [k for k in unexpected if k in {kk for kk in sub}]
print(f"loaded experts.cards.* ({len(sub)} tensors); param-level missing/unexpected: {bad if bad else 'none'}")
cards.eval()

CARD = S.TYPE_BY_NAME["card"]
NCAT = len(CARD.cat_cols)          # 6
NNUM = len(tokens.CARD_NUM)        # 19
FIELDS = NCAT + NNUM + 1           # 26 (report._state_dist card fields)


def _symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


@torch.no_grad()
def decode_card(batch):
    z = cards.encode(batch)
    return cards.decode(z)["card"]


def load_real():
    stacked, acts = D.load_fixed_sample("data/corpus", "val", VAL_STATES)
    return stacked, acts


def load_cov():
    stacked, acts = SY.coverage_val_sample(["cards"], VAL_STATES, SY.COVERAGE_VAL_SEED)
    return stacked, acts


# ==================================================================================================
# Pass 1: per-column decomposition + authoritative expert_dist reconcile, streamed in val batches.
# ==================================================================================================
def analyze(stacked, acts, label):
    n = len(acts)
    cat_mis = np.zeros(NCAT); num_mis = np.zeros(NNUM)
    kw_bit_mis = 0.0; kw_bits = 0.0; kw_row_mis = 0.0
    both_slots = 0.0; only_slots = 0.0; union_slots = 0.0
    # per-state distance (authoritative) + per-state row count
    state_num = []; state_den = []; state_rows = []
    # numeric magnitude tracking for two wide cols: damage(7), count_draw(15)
    mag = {7: {}, 15: {}}
    # abs-error histogram on NONZERO targets for key cols (tests near-miss vs gross failure / bin-res)
    abserr = {0: [0, 0, 0, 0], 7: [0, 0, 0, 0], 10: [0, 0, 0, 0], 15: [0, 0, 0, 0]}  # [=0(exact),1,2,>2]
    # collect per-row (present, both) records for downstream (dup/envelope/wildcard): store per state
    per_state_target = []   # (cats[k,6], num[k,19], kw[k,32])
    per_state_rowmis = []   # per present row: number of mismatched fields (both-present rows only; -1 if pred-absent)

    expert_dist_pairs_num = []; expert_dist_pairs_den = []
    for i in range(0, n, VAL_BATCH):
        sl = {k: stacked[k][i:i + VAL_BATCH] for k in M.BATCH_KEYS}
        batch = M.to_tensors(sl, torch.device("cpu"))
        out = decode_card(batch)
        # authoritative trainer metric (val_experts=trained-only path)
        pairs = report.report_pairs_experts_only(batch, {"card": out}, ["cards"])
        num, den = pairs["expert_dist::cards"]
        expert_dist_pairs_num.append(num); expert_dist_pairs_den.append(den)

        tmask = batch["card_mask"].bool().numpy()                       # [B,slots]
        pmask = (torch.sigmoid(out["presence"]) >= 0.5).numpy()         # [B,slots]
        both = tmask & pmask; only = tmask ^ pmask
        both_slots += both.sum(); only_slots += only.sum(); union_slots += (tmask | pmask).sum()

        tcat = batch["card_idx"].numpy()                                 # [B,slots,6]
        pcat = np.stack([out["cat"][c].argmax(-1).numpy() for c in range(NCAT)], -1)
        for c in range(NCAT):
            cat_mis[c] += ((pcat[..., c] != tcat[..., c]) & both).sum()

        tnum_i = np.round(_symexp(batch["card_num"].numpy()))           # [B,slots,19]
        pnum_i = np.round(_symexp(out["num"].numpy()))
        num_mm = (pnum_i != tnum_i)                                      # [B,slots,19]
        for j in range(NNUM):
            num_mis[j] += (num_mm[..., j] & both).sum()
        for col in mag:
            bmask = both
            tv = tnum_i[..., col][bmask].astype(int)
            mm = num_mm[..., col][bmask]
            for v, e in zip(tv, mm):
                d = mag[col].setdefault(int(v), [0, 0]); d[0] += int(e); d[1] += 1
        for col in abserr:
            bmask = both & (tnum_i[..., col] != 0)
            ae = np.abs(pnum_i[..., col][bmask] - tnum_i[..., col][bmask]).astype(int)
            h = abserr[col]
            h[0] += int((ae == 0).sum()); h[1] += int((ae == 1).sum())
            h[2] += int((ae == 2).sum()); h[3] += int((ae > 2).sum())

        tkw = (batch["card_kw"].numpy() >= 0.5)
        pkw = (torch.sigmoid(out["kw"]).numpy() >= 0.5)
        kwmm = (pkw != tkw)                                              # [B,slots,32]
        kw_bit_mis += (kwmm & both[..., None]).sum(); kw_bits += both.sum() * len(tokens.KEYWORDS)
        kw_row_mis += (kwmm.any(-1) & both).sum()

        # per-state records (for dup/envelope) — targets only, present rows
        B = tmask.shape[0]
        # per-row mismatched-field count on both-present rows (for magnitude/dup/env error attribution)
        rowmis = (np.stack([(pcat[..., c] != tcat[..., c]) for c in range(NCAT)], -1).sum(-1)
                  + num_mm.sum(-1) + kwmm.any(-1).astype(int))          # [B,slots]
        for b in range(B):
            pr = tmask[b]
            per_state_target.append((tcat[b][pr], batch["card_num"].numpy()[b][pr], tkw[b][pr]))
            rm = np.where(both[b][pr], rowmis[b][pr], -1)               # -1 => predicted absent
            per_state_rowmis.append(rm)
            state_rows.append(int(pr.sum()))
        state_num.extend(num.tolist()); state_den.extend(den.tolist())

    edist = float(np.concatenate(expert_dist_pairs_num).sum()
                  / np.concatenate(expert_dist_pairs_den).sum())
    # my decomposition total should equal edist
    my_num = only_slots * FIELDS + cat_mis.sum() + num_mis.sum() + kw_row_mis
    my_den = union_slots * FIELDS
    print(f"\n===== {label}: expert_dist::cards = {edist:.4f}  (my-decomp {my_num/my_den:.4f}, "
          f"reconcile diff={abs(edist-my_num/my_den):.2e}) =====")
    print(f"  present target slots={int(both_slots+ (only_slots if False else 0))}  "
          f"both={int(both_slots)} only(presence-err)={int(only_slots)} union={int(union_slots)}")
    print(f"  presence error share of dist = {only_slots*FIELDS/my_den:.4f}")

    # per-column table (worst first): contribution to distance = mismatches / union_slots (comparable across cols)
    rows = []
    for c in range(NCAT):
        rows.append(("CAT " + CARD.cat_cols[c][0], cat_mis[c], cat_mis[c] / both_slots,
                     cat_mis[c] / my_den))
    for j in range(NNUM):
        rng = S.NUMERIC_RANGES.get("card", {}).get(tokens.CARD_NUM[j])
        w = (rng.hi - rng.lo + 1) if rng else 2
        rows.append((f"NUM {tokens.CARD_NUM[j]}(<{w}b)", num_mis[j], num_mis[j] / both_slots,
                     num_mis[j] / my_den))
    rows.append(("KW(row-any)", kw_row_mis, kw_row_mis / both_slots, kw_row_mis / my_den))
    rows.sort(key=lambda r: -r[1])
    print(f"  {'column':28s} {'mism/both':>10s} {'dist-share':>11s}")
    for name, cnt, rate, share in rows:
        print(f"  {name:28s} {rate:10.4f} {share:11.4f}")
    print(f"  KW bit-error rate = {kw_bit_mis/max(1,kw_bits):.4f}")

    # error vs row-count-per-state
    sn = np.array(state_num); sd_ = np.array(state_den); sr = np.array(state_rows)
    print(f"  -- error vs rows-per-state --")
    edges = [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 200]
    for a, b in zip(edges, edges[1:]):
        m = (sr >= a) & (sr < b)
        if m.sum() == 0:
            continue
        d = sd_[m].sum()
        print(f"    rows[{a:>2d},{b:>3d}) states={int(m.sum()):5d} "
              f"dist={sn[m].sum()/d if d else 0:.4f}")

    # numeric magnitude (bin resolution is 1 for all card nums, so this isolates 'wrong bin' vs magnitude)
    for col in mag:
        print(f"  -- {tokens.CARD_NUM[col]} mismatch vs target magnitude --")
        items = sorted(mag[col].items())
        # bucket
        buckets = [(0, 0), (1, 3), (4, 10), (11, 30), (31, 80), (81, 999)]
        for lo, hi in buckets:
            e = sum(v[0] for k, v in items if lo <= k <= hi); t = sum(v[1] for k, v in items if lo <= k <= hi)
            if t:
                print(f"    val[{lo:>3d},{hi:>3d}] n={t:7d} miss={e/t:.4f}")
    print(f"  -- abs-error on NONZERO targets (exact / off-by-1 / off-by-2 / >2) --")
    for col, h in abserr.items():
        totn = sum(h)
        if totn:
            print(f"    {tokens.CARD_NUM[col]:14s} n={totn:6d} "
                  f"exact={h[0]/totn:.3f} +-1={h[1]/totn:.3f} +-2={h[2]/totn:.3f} >2={h[3]/totn:.3f}")
    return dict(per_state_target=per_state_target, per_state_rowmis=per_state_rowmis,
                state_rows=state_rows, edist=edist)


# ==================================================================================================
# Envelope-violation test on REAL val (does real data fall outside the generator's reachable support?)
# ==================================================================================================
def envelope_real(res):
    tbl = SY._load_reachable()
    card_ids = set(int(x) for x in tbl["card_ids"].tolist())
    id_order = {int(x): r for r, x in enumerate(tbl["card_ids"].tolist())}
    num_lo = tbl["card_num_lo"]; num_hi = tbl["card_num_hi"]     # [n_ids, n_nonzone]
    nonzone = SY._CARD_NONZONE_NUM                                # (col_in_CARD_NUM, name, is_raw)
    zone_idx = [tokens.CARD_NUM.index(z) for z in tokens.ZONE_COUNT_FIELDS]
    zone_caps = SY._CARD_ZONE_MAXES                               # hand/draw/discard/exhaust/offered
    rows_hist_max = len(SY._CARD_ROWS_HIST) - 1                   # 23
    nzones_max = int(np.nonzero(SY._CARD_NZONES_HIST)[0].max())   # 4 occupied zones max

    tot_rows = 0; tot_states = len(res["state_rows"])
    unseen_id = 0
    numcol_viol = np.zeros(len(nonzone)); anynum_viol = 0
    zonecap_viol = 0; nzone_viol = 0
    rowcount_states = 0
    # error attribution: mismatched fields on violating vs clean rows
    err_viol = [0.0, 0.0]; err_clean = [0.0, 0.0]     # [sum_mismatch_fields, n_rows] (both-present rows)
    for (cats, num, kw), rm, nrow in zip(res["per_state_target"], res["per_state_rowmis"],
                                         res["state_rows"]):
        if nrow > rows_hist_max:
            rowcount_states += 1
        for r in range(nrow):
            tot_rows += 1
            viol = False
            cid = int(cats[r, 0])
            if cid not in card_ids:
                unseen_id += 1; viol = True
            else:
                row = id_order[cid]
                for jj, (col_j, name, is_raw) in enumerate(nonzone):
                    v = num[r, col_j]
                    iv = round(float(v if is_raw else _symexp(v)))
                    if iv < num_lo[row, jj] or iv > num_hi[row, jj]:
                        numcol_viol[jj] += 1; viol = True
            # zone caps + occupied-zone count (generator support, independent of id)
            zc = np.round(_symexp(num[r, zone_idx])).astype(int)
            if (zc > zone_caps).any():
                zonecap_viol += 1; viol = True
            if int((zc > 0).sum()) > nzones_max:
                nzone_viol += 1; viol = True
            if viol:
                anynum_viol_row = True
            fields = rm[r]
            bucket = err_viol if viol else err_clean
            if fields >= 0:
                bucket[0] += fields; bucket[1] += 1
    print(f"\n===== REAL val envelope-violation vs generator reachable support =====")
    print(f"  states={tot_states} present card rows={tot_rows}")
    print(f"  rows unseen cardIndex (not in reachable table) : {unseen_id/tot_rows:.4f} "
          f"({unseen_id} rows)")
    print(f"  rows with zone-count > generator cap            : {zonecap_viol/tot_rows:.4f} "
          f"({zonecap_viol})")
    print(f"  rows with >{nzones_max} occupied zones                   : {nzone_viol/tot_rows:.4f} "
          f"({nzone_viol})")
    print(f"  states with row-count > hist max ({rows_hist_max})          : {rowcount_states/tot_states:.4f} "
          f"({rowcount_states})")
    print(f"  per-column out-of-reachable-range (in-table ids only):")
    order = np.argsort(-numcol_viol)
    for jj in order:
        if numcol_viol[jj] == 0:
            continue
        print(f"    {nonzone[jj][1]:16s} {numcol_viol[jj]/tot_rows:.4f} ({int(numcol_viol[jj])})")
    ev = err_viol[0]/max(1, err_viol[1]); ec = err_clean[0]/max(1, err_clean[1])
    print(f"  ANY-envelope-violation rows: {err_viol[1]} ({err_viol[1]/tot_rows:.4f} of rows)  "
          f"mean mismatched fields/row(of {FIELDS}) = {ev:.3f}")
    print(f"  clean (in-envelope) rows   : {err_clean[1]}  mean mismatched fields/row = {ec:.3f}")


# ==================================================================================================
# Duplicate-content-row collisions on COVERAGE (generator can emit them; real states cannot).
# ==================================================================================================
def dup_cov(res):
    n_states = len(res["per_state_target"]); dup_states = 0
    dup_rows = 0; tot_rows = 0
    err_dup = [0.0, 0.0]; err_nondup = [0.0, 0.0]
    for (cats, num, kw), rm, nrow in zip(res["per_state_target"], res["per_state_rowmis"],
                                         res["state_rows"]):
        tot_rows += nrow
        if nrow < 2:
            for r in range(nrow):
                if rm[r] >= 0:
                    err_nondup[0] += rm[r]; err_nondup[1] += 1
            continue
        cols = SY._card_content_key_columns(cats[:nrow], num[:nrow], kw[:nrow].astype(np.float32))
        key = np.stack(cols, -1)                       # [k, ncols] content key (excl zone counts)
        # find rows whose key duplicates another row in the same state
        seen = {}
        dup_flag = np.zeros(nrow, bool)
        for r in range(nrow):
            t = tuple(int(x) for x in key[r])
            if t in seen:
                dup_flag[r] = True; dup_flag[seen[t]] = True
            else:
                seen[t] = r
        if dup_flag.any():
            dup_states += 1; dup_rows += int(dup_flag.sum())
        for r in range(nrow):
            if rm[r] < 0:
                continue
            (err_dup if dup_flag[r] else err_nondup)[0] += rm[r]
            (err_dup if dup_flag[r] else err_nondup)[1] += 1
    print(f"\n===== COVERAGE duplicate-content-row collisions (generator artifact) =====")
    print(f"  states with >=1 duplicate-content row : {dup_states/n_states:.4f} ({dup_states}/{n_states})")
    print(f"  colliding rows                        : {dup_rows/max(1,tot_rows):.4f} ({dup_rows}/{tot_rows})")
    print(f"  mean mismatched fields/row  colliding = {err_dup[0]/max(1,err_dup[1]):.3f} "
          f"(n={err_dup[1]})  vs non-colliding = {err_nondup[0]/max(1,err_nondup[1]):.3f} (n={err_nondup[1]})")


# ==================================================================================================
# Wildcard-row share of COVERAGE error (rows whose cardIndex is outside the reachable id set).
# ==================================================================================================
def wildcard_cov(res):
    tbl = SY._load_reachable()
    card_ids = set(int(x) for x in tbl["card_ids"].tolist())
    err_w = [0.0, 0.0]; err_t = [0.0, 0.0]; tot = 0
    for (cats, num, kw), rm, nrow in zip(res["per_state_target"], res["per_state_rowmis"],
                                         res["state_rows"]):
        for r in range(nrow):
            tot += 1
            if rm[r] < 0:
                continue
            w = int(cats[r, 0]) not in card_ids
            (err_w if w else err_t)[0] += rm[r]; (err_w if w else err_t)[1] += 1
    tot_err = err_w[0] + err_t[0]
    print(f"\n===== COVERAGE wildcard-id rows (cardIndex outside reachable set) =====")
    print(f"  wildcard-id rows share      : {err_w[1]/max(1,err_w[1]+err_t[1]):.4f}")
    print(f"  mean mismatched fields/row  wildcard = {err_w[0]/max(1,err_w[1]):.3f}  "
          f"table = {err_t[0]/max(1,err_t[1]):.3f}")
    print(f"  wildcard share of TOTAL mismatched fields : {err_w[0]/max(1,tot_err):.4f}")


if __name__ == "__main__":
    print("\n########## REAL VAL ##########")
    rs, ra = load_real()
    real = analyze(rs, ra, "REAL")
    envelope_real(real)
    print("\n########## COVERAGE VAL ##########")
    cs, ca = load_cov()
    cov = analyze(cs, ca, "COVERAGE")
    dup_cov(cov)
    wildcard_cov(cov)
