"""Scratch diagnostic (measurement only, no product changes): what holds the STATIC-ONLY cards-expert
plateau (checkpoint cards_static.pt(.best), trained with --cards-static-only: the loss+report mask every
transient CARD_NUM column, keeping only `upgraded`; commit 3ad967c).

The probe sits at real dist ~0.033 / exact ~0.45 but coverage dist ~0.104 / exact ~0.0045 — the SYNTHETIC
space fails on the STATIC fields. This script applies the SAME static-only field mask the run's report uses
(report._state_dist(..., cards_static_only=True): 8 cat cols + `upgraded` numeric + the 32-bit keyword
block + presence == 10 scored fields/slot) so the numbers reconcile to the recorded dist first, then
decomposes the coverage residual, testing the LEADING HYPOTHESIS:

    the generator's 5% per-row WILDCARD tail (SY.CARD_WILDCARD_PROB) — uniform cardIndex over the whole
    vocab, uniform enchant/afflict over 128 buckets, random-sparse 32-bit keywords, incompressible noise
    absent from real states — dominates the coverage residual.

CPU only (a GPU run is in progress; never allocate CUDA). Python 3.9. scipy not required.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # never touch the GPU
import numpy as np
import torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lts2_agent import tokens, catalog
from lts2_agent.wm import spec as S, model_factored as MF, report, data as D, synth as SY
from lts2_agent.wm import model as M
from lts2_agent.wm.experts import CARD_STATIC_NUM_KEEP

CKPT_BASE = "C:/Users/Alexe/.claude/jobs/06eda07b/tmp/cards_static.pt"
CKPT = CKPT_BASE + ".best" if os.path.exists(CKPT_BASE + ".best") else CKPT_BASE
VAL_STATES = 2000
VAL_BATCH = 256
CORPUS = "data/corpus"

CARD = S.TYPE_BY_NAME["card"]
CAT_NAMES = [c for c, _ in CARD.cat_cols]          # 8: cardIndex,type,rarity,targetType,enchant,afflict,zone,slot
NCAT = len(CAT_NAMES)                               # 8
NUM_NAMES = list(tokens.CARD_NUM)                   # 14
STATIC_NUM_COLS = list(CARD_STATIC_NUM_KEEP)        # [3] == upgraded
STATIC_NUM_NAMES = [NUM_NAMES[j] for j in STATIC_NUM_COLS]
NSNUM = len(STATIC_NUM_COLS)                        # 1
# Static-masked scored fields per present slot: 8 cats + kept nums + 1 keyword block.
FIELDS = NCAT + NSNUM + 1                           # 10
ZONE_COL = CAT_NAMES.index("zone")                  # 6
SLOT_COL = CAT_NAMES.index("slot")                  # 7
CID_COL = CAT_NAMES.index("cardIndex")              # 0
ENCH_COL = CAT_NAMES.index("enchant")               # 4
AFFL_COL = CAT_NAMES.index("afflict")               # 5
ZONES = tokens.ZONES
KWB = len(tokens.KEYWORDS)
_CARD_CAT = catalog.load("cards")


def _symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


# ==================================================================================================
# Reachable-table observed sets — the ground truth for wildcard/table classification of a target row.
# A row is WILDCARD-LIKE if its cardIndex is outside the reachable table's ids, OR its enchant/afflict
# value is outside that card's observed values, OR its keyword pattern is not one of that card's observed
# patterns. (Table rows always draw from these sets, so they never trip; a generator-wildcard row that
# coincidentally lands entirely inside the table is indistinguishable from a table row — and equally
# learnable — so leaving it labeled "table" is the conservative, correct choice.)
# ==================================================================================================
def build_observed():
    tbl = SY._load_reachable()
    cards = tbl["cards"]
    reachable_ids = set(int(k) for k in cards.keys())
    ench_sets = {}
    affl_sets = {}
    kw_sets = {}
    kw_canon = {}    # cid -> multi-hot union? no; keep list of observed patterns for "canonical" test
    for cid, e in cards.items():
        ench_sets[int(cid)] = set(int(v) for v in e["enchant_vals"].tolist())
        affl_sets[int(cid)] = set(int(v) for v in e["afflict_vals"].tolist())
        kw_sets[int(cid)] = set(tuple(sorted(int(b) for b in pat)) for pat in e["keywords"])
        kw_canon[int(cid)] = list(kw_sets[int(cid)])
    return dict(reachable_ids=reachable_ids, ench=ench_sets, affl=affl_sets, kw=kw_sets, kw_canon=kw_canon)


def classify_row(obs, cid, ench, affl, kw_pat):
    """Return (is_wildcard, reason) for a single decoded target row."""
    if cid not in obs["reachable_ids"]:
        return True, "id"
    if ench not in obs["ench"][cid]:
        return True, "enchant"
    if affl not in obs["affl"][cid]:
        return True, "afflict"
    if kw_pat not in obs["kw"][cid]:
        return True, "keyword"
    return False, "table"


# ==================================================================================================
# Load per-state decoded arrays (target + predicted), aligned by slot position, + authoritative dist.
# ==================================================================================================
def load_states(model, loader):
    stacked, acts = loader()
    n = len(acts)
    states = []
    auth_num = []
    auth_den = []
    for i in range(0, n, VAL_BATCH):
        sl = {k: stacked[k][i:i + VAL_BATCH] for k in M.BATCH_KEYS}
        batch = M.to_tensors(sl, torch.device("cpu"))
        with torch.no_grad():
            _, outputs = model.forward(batch, active_experts=["cards"])
        pairs = report.report_pairs_experts_only(batch, outputs, ["cards"], cards_static_only=True)
        an, ad = pairs["expert_dist::cards"]
        auth_num.append(an)
        auth_den.append(ad)
        B = batch["global_idx"].shape[0]
        pred = report._reconstruct_types(outputs, [CARD], B)
        tgt = report._target_arrays(batch)
        for b in range(B):
            tm = tgt[b]["card_mask"].astype(bool)
            pm = pred[b]["card_mask"].astype(bool)
            tcat = np.atleast_2d(tgt[b]["card_idx"]).astype(np.int64)
            pcat = np.atleast_2d(pred[b]["card_idx"]).astype(np.int64)
            tnum = np.round(_symexp(np.atleast_2d(tgt[b]["card_num"]).astype(np.float64))).astype(np.int64)
            pnum = np.round(_symexp(np.atleast_2d(pred[b]["card_num"]).astype(np.float64))).astype(np.int64)
            tkw = (np.atleast_2d(tgt[b]["card_kw"]) >= 0.5)
            pkw = (np.atleast_2d(pred[b]["card_kw"]) >= 0.5)
            states.append(dict(tm=tm, pm=pm, tcat=tcat, pcat=pcat, tnum=tnum, pnum=pnum, tkw=tkw, pkw=pkw))
    auth = float(np.concatenate(auth_num).sum() / np.concatenate(auth_den).sum())
    return states, auth


# ==================================================================================================
# Static-masked per-slot numerator. mismatched fields for a set of both-present rows + the presence cost.
# ==================================================================================================
def _row_field_mismatch(s, rows):
    """Per-row (over `rows` indices) count of mismatched STATIC fields (max FIELDS), both-present assumed."""
    tc = s["tcat"][rows]; pc = s["pcat"][rows]
    tn = s["tnum"][rows][:, STATIC_NUM_COLS]; pn = s["pnum"][rows][:, STATIC_NUM_COLS]
    tk = s["tkw"][rows]; pk = s["pkw"][rows]
    e = (pc != tc).sum(1) + (pn != tn).sum(1) + (pk != tk).any(1).astype(np.int64)
    return e


# ==================================================================================================
# M1: per-column mismatch table (static-masked, slot-matched), + reconcile to authoritative expert_dist.
# ==================================================================================================
def m1_columns(states, label, auth):
    cat_mis = np.zeros(NCAT)
    snum_mis = np.zeros(NSNUM)
    kw_row_mis = 0.0
    kw_bit_mis = 0.0
    kw_bits = 0.0
    kw_bit_mis_vec = np.zeros(KWB)
    both_slots = 0.0
    only_slots = 0.0
    union_slots = 0.0
    exact_states = 0
    for s in states:
        both = s["tm"] & s["pm"]
        only = s["tm"] ^ s["pm"]
        both_slots += both.sum(); only_slots += only.sum(); union_slots += (s["tm"] | s["pm"]).sum()
        st_num = only.sum() * FIELDS
        if both.any():
            st_num += _row_field_mismatch(s, np.nonzero(both)[0]).sum()
        if st_num == 0:
            exact_states += 1
        if not both.any():
            continue
        for c in range(NCAT):
            cat_mis[c] += ((s["pcat"][:, c] != s["tcat"][:, c]) & both).sum()
        nmm = (s["pnum"][:, STATIC_NUM_COLS] != s["tnum"][:, STATIC_NUM_COLS])
        for j in range(NSNUM):
            snum_mis[j] += (nmm[:, j] & both).sum()
        kwmm = (s["pkw"] != s["tkw"])
        kw_row_mis += (kwmm.any(-1) & both).sum()
        kw_bit_mis += (kwmm & both[:, None]).sum()
        kw_bit_mis_vec += (kwmm & both[:, None]).sum(0)
        kw_bits += both.sum() * KWB
    my_num = only_slots * FIELDS + cat_mis.sum() + snum_mis.sum() + kw_row_mis
    my_den = union_slots * FIELDS
    my = my_num / my_den
    print(f"\n===== M1 {label}: expert_dist::cards[static] = {auth:.4f}  (decomp {my:.4f}, "
          f"reconcile diff {abs(auth-my):.2e})   state-exact-rate = "
          f"{exact_states/max(1,len(states)):.4f} =====")
    print(f"  present target slots both={int(both_slots)} presence-err(only)={int(only_slots)} "
          f"union={int(union_slots)}   [FIELDS/slot={FIELDS}]")
    rows = []
    for c in range(NCAT):
        rows.append(("CAT " + CAT_NAMES[c], cat_mis[c] / max(1, both_slots), cat_mis[c] / my_den))
    for j in range(NSNUM):
        rows.append(("NUM " + STATIC_NUM_NAMES[j], snum_mis[j] / max(1, both_slots), snum_mis[j] / my_den))
    rows.append(("KW(row-any)", kw_row_mis / max(1, both_slots), kw_row_mis / my_den))
    rows.append(("PRESENCE", only_slots / max(1, union_slots), only_slots * FIELDS / my_den))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'column':22s} {'rate/both':>10s} {'dist-share':>11s}")
    for name, rate, share in rows:
        print(f"  {name:22s} {rate:10.4f} {share:11.4f}")
    print(f"  KW bit-error rate (over both slots x {KWB} bits) = {kw_bit_mis/max(1,kw_bits):.4f}")
    worst_bits = np.argsort(-kw_bit_mis_vec)[:6]
    print("  KW worst bits (bit: err count): " +
          ", ".join(f"{int(b)}:{int(kw_bit_mis_vec[b])}" for b in worst_bits))
    return dict(both=both_slots, only=only_slots, union=union_slots, my_den=my_den)


# ==================================================================================================
# M2: WILDCARD decomposition (coverage). Classify each present TARGET row wildcard-like vs table-like.
# ==================================================================================================
def m2_wildcard(states, obs, label):
    # per-class tallies
    n_rows = {"wild": 0, "table": 0}
    err_rows = {"wild": 0.0, "table": 0.0}          # summed mismatched static fields (present target rows)
    reason_ct = {"id": 0, "enchant": 0, "afflict": 0, "keyword": 0}
    # state-level: exact rate for pure-table states vs states with >=1 wildcard target row
    states_with_wild = 0
    n_states = 0
    exact_wildstate = [0, 0]     # [exact, total]
    exact_puretable = [0, 0]
    # total coverage error accounting (== auth numerator): target-row error + spurious-pred error
    tot_num = 0.0
    tot_den = 0.0
    spurious_num = 0.0           # pred-present, target-absent slots (10 each) — cannot be class-attributed
    target_missing_num = {"wild": 0.0, "table": 0.0}   # target-present pred-absent rows (10 each)

    for s in states:
        n_states += 1
        tm = s["tm"]; pm = s["pm"]
        both = tm & pm
        only = tm ^ pm
        tgt_only = tm & (~pm)      # target present, pred absent
        prd_only = pm & (~tm)      # pred present, target absent (spurious)
        tot_den += (both.sum() + only.sum()) * FIELDS
        spurious_num += prd_only.sum() * FIELDS
        # state numerator (static) for exact-rate
        st_num = only.sum() * FIELDS
        if both.any():
            st_num += _row_field_mismatch(s, np.nonzero(both)[0]).sum()
        tot_num += st_num + 0.0    # spurious already in only.sum(); careful: only includes both tgt_only+prd_only
        # classify each present TARGET row
        T = np.nonzero(tm)[0]
        has_wild = False
        for r in T:
            cid = int(s["tcat"][r, CID_COL])
            ench = int(s["tcat"][r, ENCH_COL])
            affl = int(s["tcat"][r, AFFL_COL])
            pat = tuple(sorted(int(b) for b in np.nonzero(s["tkw"][r])[0]))
            is_wild, reason = classify_row(obs, cid, ench, affl, pat)
            cls = "wild" if is_wild else "table"
            n_rows[cls] += 1
            if is_wild:
                has_wild = True
                reason_ct[reason] += 1
            # this target row's contribution to the numerator
            if pm[r]:
                e = float(_row_field_mismatch(s, np.array([r]))[0])
                err_rows[cls] += e
            else:
                err_rows[cls] += FIELDS
                target_missing_num[cls] += FIELDS
        if has_wild:
            states_with_wild += 1
        bucket = exact_wildstate if has_wild else exact_puretable
        bucket[1] += 1
        if st_num == 0.0:
            bucket[0] += 1

    tot_rows = n_rows["wild"] + n_rows["table"]
    # target-attributable error total
    tgt_err_tot = err_rows["wild"] + err_rows["table"]
    print(f"\n===== M2 {label}: WILDCARD decomposition (present TARGET rows; static field set) =====")
    print(f"  total present target rows = {tot_rows}")
    for cls in ("wild", "table"):
        pct = 100.0 * n_rows[cls] / max(1, tot_rows)
        mean_e = err_rows[cls] / max(1, n_rows[cls])
        print(f"  {cls:6s}: {n_rows[cls]:7d} rows ({pct:5.1f}%)  mean mismatched fields/row = {mean_e:.3f}")
    print(f"  wildcard trip reasons (a row may trip several; first-hit order id>ench>affl>kw): "
          f"{reason_ct}")
    print(f"  --- share of TOTAL coverage error (auth numerator = {tot_num:.0f}, den = {tot_den:.0f}, "
          f"dist = {tot_num/max(1,tot_den):.4f}) ---")
    print(f"    target-attributable error   = {tgt_err_tot:.0f} "
          f"({100.0*tgt_err_tot/max(1,tot_num):.1f}% of total)")
    print(f"      of which WILDCARD rows     = {err_rows['wild']:.0f} "
          f"({100.0*err_rows['wild']/max(1,tot_num):.1f}% of TOTAL, "
          f"{100.0*err_rows['wild']/max(1,tgt_err_tot):.1f}% of target-attributable)")
    print(f"      of which TABLE rows        = {err_rows['table']:.0f} "
          f"({100.0*err_rows['table']/max(1,tot_num):.1f}% of TOTAL, "
          f"{100.0*err_rows['table']/max(1,tgt_err_tot):.1f}% of target-attributable)")
    print(f"    spurious pred-only rows      = {spurious_num:.0f} "
          f"({100.0*spurious_num/max(1,tot_num):.1f}% of total)")
    print(f"  --- states ---")
    print(f"    states with >=1 wildcard target row = {100.0*states_with_wild/max(1,n_states):.1f}%")
    print(f"    exact-rate  wildcard-containing states = "
          f"{exact_wildstate[0]/max(1,exact_wildstate[1]):.4f} (n={exact_wildstate[1]})")
    print(f"    exact-rate  PURE-TABLE states          = "
          f"{exact_puretable[0]/max(1,exact_puretable[1]):.4f} (n={exact_puretable[1]})")
    return dict(n_rows=n_rows, err_rows=err_rows, tot_num=tot_num,
                exact_pure=exact_puretable, exact_wild=exact_wildstate)


# ==================================================================================================
# M3: keyword analysis — observed-pattern rows vs random-bit rows; per-bit; canonical-vs-target pred.
# ==================================================================================================
def m3_keywords(states, obs, label):
    # split present both-rows by whether the TARGET kw pattern is observed for its (reachable) card
    obs_rows = {"n": 0, "row_err": 0, "bit_err": 0, "bits": 0}
    rnd_rows = {"n": 0, "row_err": 0, "bit_err": 0, "bits": 0}
    obs_bitvec = np.zeros(KWB); rnd_bitvec = np.zeros(KWB)
    # when target kw is a random/wildcard pattern (id reachable), does pred match the CARD's canonical
    # observed pattern set instead of the noisy target?
    pred_is_canon = 0     # pred kw pattern is one of the card's observed patterns
    pred_is_target = 0    # pred kw pattern == the (random) target
    rnd_reach = 0
    for s in states:
        both = s["tm"] & s["pm"]
        for r in np.nonzero(both)[0]:
            cid = int(s["tcat"][r, CID_COL])
            tk = s["tkw"][r]; pk = s["pkw"][r]
            pat = tuple(sorted(int(b) for b in np.nonzero(tk)[0]))
            reachable = cid in obs["reachable_ids"]
            is_obs = reachable and (pat in obs["kw"][cid])
            bucket = obs_rows if is_obs else rnd_rows
            bitvec = obs_bitvec if is_obs else rnd_bitvec
            bucket["n"] += 1
            bucket["row_err"] += int((pk != tk).any())
            be = (pk != tk)
            bucket["bit_err"] += int(be.sum())
            bucket["bits"] += KWB
            bitvec += be
            if (not is_obs) and reachable:
                rnd_reach += 1
                ppat = tuple(sorted(int(b) for b in np.nonzero(pk)[0]))
                if ppat in obs["kw"][cid]:
                    pred_is_canon += 1
                if ppat == pat:
                    pred_is_target += 1
    print(f"\n===== M3 {label}: keyword analysis =====")
    for name, d, bv in (("observed-pattern", obs_rows, obs_bitvec), ("random-bit", rnd_rows, rnd_bitvec)):
        if d["n"]:
            print(f"  {name:16s} rows n={d['n']:7d}  row-any-err={d['row_err']/d['n']:.4f}  "
                  f"per-bit-err={d['bit_err']/max(1,d['bits']):.4f}")
    if rnd_reach:
        print(f"  random-kw rows w/ REACHABLE id (n={rnd_reach}): "
              f"pred matches card's CANONICAL pattern set = {100.0*pred_is_canon/rnd_reach:.1f}%   "
              f"pred matches the (noise) TARGET = {100.0*pred_is_target/rnd_reach:.1f}%")


# ==================================================================================================
# M4: enchant / afflict — observed vs wildcard split.
# ==================================================================================================
def m4_enchant_afflict(states, obs, label):
    for col, name, oset_key in ((ENCH_COL, "enchant", "ench"), (AFFL_COL, "afflict", "affl")):
        obs_n = 0; obs_err = 0; wild_n = 0; wild_err = 0
        wild_pred_obs = 0     # target value is off-table but pred lands on an observed value
        for s in states:
            both = s["tm"] & s["pm"]
            for r in np.nonzero(both)[0]:
                cid = int(s["tcat"][r, CID_COL])
                tv = int(s["tcat"][r, col]); pv = int(s["pcat"][r, col])
                reachable = cid in obs["reachable_ids"]
                on_table = reachable and (tv in obs[oset_key][cid])
                if on_table:
                    obs_n += 1; obs_err += int(pv != tv)
                else:
                    wild_n += 1; wild_err += int(pv != tv)
                    if reachable and (pv in obs[oset_key][cid]):
                        wild_pred_obs += 1
        print(f"\n===== M4 {label}: {name} observed-vs-wildcard =====")
        if obs_n:
            print(f"  observed-value rows n={obs_n:7d}  err={obs_err/obs_n:.4f}")
        if wild_n:
            print(f"  wildcard-value rows n={wild_n:7d}  err={wild_err/wild_n:.4f}   "
                  f"pred lands on an OBSERVED value (reachable id) = "
                  f"{100.0*wild_pred_obs/max(1,wild_n):.1f}%")


# ==================================================================================================
# M5: cardIndex — hallucination vs in-state confusion; error vs rows-per-state; large-deck tail.
# ==================================================================================================
def m5_cardindex(states, label):
    miss_by_id = {}
    wrong_in_state = 0
    wrong_absent = 0
    # error vs rows-per-state (mean mismatched STATIC fields/row) + large-deck tail share
    redges = [0, 4, 8, 12, 16, 20, 24, 200]
    rbuck = {i: [0.0, 0] for i in range(len(redges) - 1)}
    tail_err = 0.0
    tot_err = 0.0
    for s in states:
        both = s["tm"] & s["pm"]
        pos = np.nonzero(both)[0]
        T = np.nonzero(s["tm"])[0]
        k = len(T)
        # per-row static error for present target rows (pred-absent -> FIELDS)
        errs = np.zeros(k)
        for kk, r in enumerate(T):
            if s["pm"][r]:
                errs[kk] = _row_field_mismatch(s, np.array([r]))[0]
            else:
                errs[kk] = FIELDS
        if k:
            ri = np.searchsorted(redges, k, side="right") - 1
            ri = min(ri, len(redges) - 2)
            rbuck[ri][0] += errs.sum(); rbuck[ri][1] += k
            tot_err += errs.sum()
            if k >= 16:
                tail_err += errs.sum()
        # cardIndex confusion on both-present rows
        if len(pos):
            tid = s["tcat"][pos, CID_COL]; pid = s["pcat"][pos, CID_COL]
            present_ids = set(int(x) for x in s["tcat"][T, CID_COL])
            for t, p in zip(tid, pid):
                if t == p:
                    continue
                miss_by_id[int(t)] = miss_by_id.get(int(t), 0) + 1
                if int(p) in present_ids:
                    wrong_in_state += 1
                else:
                    wrong_absent += 1
    tot = wrong_in_state + wrong_absent
    print(f"\n===== M5 {label}: cardIndex confusion + structure =====")
    print(f"  cardIndex mismatches (both-present) = {tot}")
    if tot:
        print(f"    wrong id present ELSEWHERE in state (copy/slot confusion) = {wrong_in_state} "
              f"({100.0*wrong_in_state/tot:.1f}%)")
        print(f"    wrong id ABSENT from state (true hallucination)          = {wrong_absent} "
              f"({100.0*wrong_absent/tot:.1f}%)")
    print(f"  -- static err/row by rows-per-state --")
    for i in range(len(redges) - 1):
        d = rbuck[i]
        print(f"    [{redges[i]:2d},{redges[i+1]:3d})  n={d[1]:7d}  err/row={d[0]/max(1,d[1]):.3f}")
    print(f"  large-deck tail (rows>=16) share of total static row-error = "
          f"{100.0*tail_err/max(1,tot_err):.1f}%")
    print(f"  top-10 most-missed cardIndex:")
    for cid, cnt in sorted(miss_by_id.items(), key=lambda kv: -kv[1])[:10]:
        nm = _CARD_CAT.id_of(cid) or "(none/id0)"
        print(f"    id={cid:4d} miss={cnt:5d}  {nm}")


# ==================================================================================================
# M6: REAL-val drivers — per-zone err/row, err vs rows-per-state, top-10 missed ids.
# ==================================================================================================
def m6_real(states, label):
    print(f"\n===== M6 {label}: real-val error structure =====")
    zbuck = {z: [0.0, 0] for z in range(len(ZONES))}
    redges = [0, 4, 8, 12, 16, 20, 200]
    rbuck = {i: [0.0, 0] for i in range(len(redges) - 1)}
    for s in states:
        T = np.nonzero(s["tm"])[0]
        k = len(T)
        if k == 0:
            continue
        errs = np.zeros(k)
        for kk, r in enumerate(T):
            errs[kk] = _row_field_mismatch(s, np.array([r]))[0] if s["pm"][r] else FIELDS
        zones = s["tcat"][T, ZONE_COL]
        for z, e in zip(zones, errs):
            if 0 <= z < len(ZONES):
                zbuck[int(z)][0] += e; zbuck[int(z)][1] += 1
        ri = np.searchsorted(redges, k, side="right") - 1
        ri = min(ri, len(redges) - 2)
        rbuck[ri][0] += errs.sum(); rbuck[ri][1] += k
    print("  -- static err/row by zone --")
    for z in range(len(ZONES)):
        d = zbuck[z]
        print(f"    {ZONES[z]:10s} n={d[1]:7d}  err/row={d[0]/max(1,d[1]):.3f}")
    print("  -- static err/row by rows-per-state --")
    for i in range(len(redges) - 1):
        d = rbuck[i]
        print(f"    [{redges[i]:2d},{redges[i+1]:3d})  n={d[1]:7d}  err/row={d[0]/max(1,d[1]):.3f}")


def main():
    print(f"ckpt={CKPT}")
    model, meta = MF.load_checkpoint(CKPT, device="cpu")
    model.eval()
    print(f"step={meta['step']} num_input={model.num_input} "
          f"best_state_dist={meta.get('best_state_dist'):.4f}")
    print(f"static field set: {NCAT} cats {CAT_NAMES}  + num {STATIC_NUM_NAMES}  + kw + presence "
          f"= {FIELDS} fields/slot")

    obs = build_observed()
    print(f"reachable card ids: {len(obs['reachable_ids'])}")

    # REAL
    print("\n" + "#" * 96 + "\n########## REAL ##########")
    rstates, rauth = load_states(model, lambda: D.load_fixed_sample(CORPUS, "val", VAL_STATES))
    m1_columns(rstates, "REAL", rauth)
    m5_cardindex(rstates, "REAL")
    m6_real(rstates, "REAL")

    # COVERAGE
    print("\n" + "#" * 96 + "\n########## COVERAGE ##########")
    cstates, cauth = load_states(
        model, lambda: SY.coverage_val_sample(["cards"], VAL_STATES, SY.COVERAGE_VAL_SEED))
    m1_columns(cstates, "COVERAGE", cauth)
    m2_wildcard(cstates, obs, "COVERAGE")
    m3_keywords(cstates, obs, "COVERAGE")
    m4_enchant_afflict(cstates, obs, "COVERAGE")
    m5_cardindex(cstates, "COVERAGE")


if __name__ == "__main__":
    main()
