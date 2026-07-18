"""Scratch diagnostic (measurement only, no product changes): what holds the cards-expert plateau
under tokenizer v6 (instance-anchored rows: one row per physical card copy, laid out ZONE-MAJOR and
content-sorted within a zone; ``slot`` == the row's index in that layout).

Loads the full factored v6 checkpoint (cards_v6_50k.pt(.best)) and drives ONLY the cards expert on the
same real fixed val (2000) + coverage fixed sample (2000) the trainer uses. Reconciles slot-matched
expert_dist::cards to the recorded ~0.057 real / ~0.095 coverage first, then decomposes the plateau.

Evaluates the owner's hypotheses:
  H1a  residual ORDER unpredictability from the ZONE-MAJOR layout: slot k's content depends on the
       SIZES of every earlier zone -> cross-zone shifting.
  H1b  within-zone content-sort tie-fallthrough between same-cardIndex copies that differ only in
       volatile numerics.
  H2   split "behaviour" (content) from "position" (zone/slot) into different encodings/experts.

CPU only (a GPU run is in progress; never allocate CUDA). Python 3.9, scipy used if present.
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

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

CKPT = "C:/Users/Alexe/.claude/jobs/06eda07b/tmp/cards_v6_50k.pt.best"
VAL_STATES = 2000
VAL_BATCH = 256
CORPUS = "data/corpus"

CARD = S.TYPE_BY_NAME["card"]
CAT_NAMES = [c for c, _ in CARD.cat_cols]          # 8: cardIndex..afflict, zone, slot
NCAT = len(CAT_NAMES)                               # 8
NUM_NAMES = list(tokens.CARD_NUM)                   # 14
NNUM = len(NUM_NAMES)                               # 14
FIELDS = NCAT + NNUM + 1                            # 23 (cats + nums + kw block)
ZONE_COL = CAT_NAMES.index("zone")                  # 6
SLOT_COL = CAT_NAMES.index("slot")                  # 7
CID_COL = CAT_NAMES.index("cardIndex")              # 0
ZONES = tokens.ZONES
_CARD_CAT = catalog.load("cards")


def _symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


def _greedy_assignment(cost):
    a, b = cost.shape
    order = np.argsort(cost, axis=None)
    used_r = np.zeros(a, bool); used_c = np.zeros(b, bool)
    ri = []; ci = []
    for flat in order:
        r, c = divmod(int(flat), b)
        if used_r[r] or used_c[c]:
            continue
        used_r[r] = used_c[c] = True
        ri.append(r); ci.append(c)
        if len(ri) == min(a, b):
            break
    o = np.argsort(ri)
    return np.array(ri)[o], np.array(ci)[o]


def _assign(cost):
    if HAVE_SCIPY:
        return linear_sum_assignment(cost)
    return _greedy_assignment(cost)


# ==================================================================================================
# Load per-state decoded arrays (target + predicted), aligned by slot position.
# ==================================================================================================
def load_states(model, loader):
    stacked, acts = loader()
    n = len(acts)
    states = []               # per state: dict of numpy arrays (target + pred, decoded ints)
    auth_num = []; auth_den = []
    for i in range(0, n, VAL_BATCH):
        sl = {k: stacked[k][i:i + VAL_BATCH] for k in M.BATCH_KEYS}
        batch = M.to_tensors(sl, torch.device("cpu"))
        with torch.no_grad():
            _, outputs = model.forward(batch, active_experts=["cards"])
        pairs = report.report_pairs_experts_only(batch, outputs, ["cards"])
        an, ad = pairs["expert_dist::cards"]
        auth_num.append(an); auth_den.append(ad)
        B = batch["global_idx"].shape[0]
        pred = report._reconstruct_types(outputs, [CARD], B)
        tgt = report._target_arrays(batch)
        for b in range(B):
            tm = tgt[b]["card_mask"].astype(bool)
            pm = pred[b]["card_mask"].astype(bool)
            tcat = np.atleast_2d(tgt[b]["card_idx"]).astype(np.int64)      # [64,8]
            pcat = np.atleast_2d(pred[b]["card_idx"]).astype(np.int64)
            tnum = np.round(_symexp(np.atleast_2d(tgt[b]["card_num"]).astype(np.float64))).astype(np.int64)
            pnum = np.round(_symexp(np.atleast_2d(pred[b]["card_num"]).astype(np.float64))).astype(np.int64)
            tkw = (np.atleast_2d(tgt[b]["card_kw"]) >= 0.5)
            pkw = (np.atleast_2d(pred[b]["card_kw"]) >= 0.5)
            states.append(dict(tm=tm, pm=pm, tcat=tcat, pcat=pcat, tnum=tnum, pnum=pnum,
                               tkw=tkw, pkw=pkw))
    auth = float(np.concatenate(auth_num).sum() / np.concatenate(auth_den).sum())
    return states, auth


# ==================================================================================================
# M1: per-column mismatch table (slot-matched), + reconcile to authoritative expert_dist.
# ==================================================================================================
def m1_columns(states, label, auth):
    cat_mis = np.zeros(NCAT); num_mis = np.zeros(NNUM)
    kw_row_mis = 0.0; kw_bit_mis = 0.0; kw_bits = 0.0
    both_slots = 0.0; only_slots = 0.0; union_slots = 0.0
    for s in states:
        both = s["tm"] & s["pm"]; only = s["tm"] ^ s["pm"]
        both_slots += both.sum(); only_slots += only.sum(); union_slots += (s["tm"] | s["pm"]).sum()
        if not both.any():
            continue
        for c in range(NCAT):
            cat_mis[c] += ((s["pcat"][:, c] != s["tcat"][:, c]) & both).sum()
        nmm = (s["pnum"] != s["tnum"])
        for j in range(NNUM):
            num_mis[j] += (nmm[:, j] & both).sum()
        kwmm = (s["pkw"] != s["tkw"])
        kw_row_mis += (kwmm.any(-1) & both).sum()
        kw_bit_mis += (kwmm & both[:, None]).sum(); kw_bits += both.sum() * len(tokens.KEYWORDS)
    my_num = only_slots * FIELDS + cat_mis.sum() + num_mis.sum() + kw_row_mis
    my_den = union_slots * FIELDS
    my = my_num / my_den
    print(f"\n===== M1 {label}: expert_dist::cards = {auth:.4f}  (decomp {my:.4f}, "
          f"reconcile diff {abs(auth-my):.2e}) =====")
    print(f"  present target slots both={int(both_slots)} presence-err(only)={int(only_slots)} "
          f"union={int(union_slots)}")
    print(f"  presence share of dist = {only_slots*FIELDS/my_den:.4f}")
    rows = []
    for c in range(NCAT):
        rows.append(("CAT " + CAT_NAMES[c], cat_mis[c], cat_mis[c] / max(1, both_slots),
                     cat_mis[c] / my_den))
    for j in range(NNUM):
        rows.append(("NUM " + NUM_NAMES[j], num_mis[j], num_mis[j] / max(1, both_slots),
                     num_mis[j] / my_den))
    rows.append(("KW(row-any)", kw_row_mis, kw_row_mis / max(1, both_slots), kw_row_mis / my_den))
    rows.append(("PRESENCE", only_slots * FIELDS, only_slots / max(1, union_slots),
                 only_slots * FIELDS / my_den))
    rows.sort(key=lambda r: -r[3])
    print(f"  {'column':22s} {'rate/both':>10s} {'dist-share':>11s}")
    for name, cnt, rate, share in rows:
        print(f"  {name:22s} {rate:10.4f} {share:11.4f}")
    print(f"  KW bit-error rate = {kw_bit_mis/max(1,kw_bits):.4f}")
    return dict(both=both_slots, only=only_slots, union=union_slots, my_den=my_den,
                cat_mis=cat_mis, num_mis=num_mis, kw_row_mis=kw_row_mis)


# ==================================================================================================
# M2: Hungarian content-matched dist (cost EXCLUDES the positional slot column). with-zone /
# without-zone variants. Alignment gap, % non-identity, cross-zone vs within-zone bucketing.
# ==================================================================================================
def _cost(tcat, tnum, tkw, pcat, pnum, pkw, cat_cols):
    a = tcat.shape[0]; b = pcat.shape[0]
    cost = np.zeros((a, b), np.float64)
    for c in cat_cols:
        cost += (tcat[:, c][:, None] != pcat[:, c][None, :])
    cost += (tnum[:, None, :] != pnum[None, :, :]).sum(-1)
    cost += (tkw[:, None, :] != pkw[None, :, :]).any(-1)
    return cost


def _slot_num_state(s):
    """Slot-matched numerator + denominator for one state (matches report._state_dist card fields)."""
    both = s["tm"] & s["pm"]; only = s["tm"] ^ s["pm"]
    den = (both.sum() + only.sum()) * FIELDS
    num = only.sum() * FIELDS
    if both.any():
        num += (s["pcat"][both] != s["tcat"][both]).sum()
        num += (s["pnum"][both] != s["tnum"][both]).sum()
        num += ((s["pkw"][both] != s["tkw"][both]).any(-1)).sum()
    return float(num), float(den)


def m2_content_match(states, label):
    # Each variant scores over EXACTLY the fields in its cost (slot always excluded; without-zone also
    # excludes zone). Slot-matched is scored over the SAME field set, so gap = pure alignment win and is
    # >= 0 by construction (identity is a feasible assignment). nf = fields scored for the variant.
    variants = {"with-zone": [c for c in range(NCAT) if c != SLOT_COL],
                "without-zone": [c for c in range(NCAT) if c not in (SLOT_COL, ZONE_COL)]}
    res = {}
    for vname, cat_cols in variants.items():
        nf = len(cat_cols) + NNUM + 1        # scored fields for this variant
        content_num_tot = 0.0; slot_num_tot = 0.0; den_tot = 0.0
        nonident = 0; nstates = 0
        cross_zone_pairs = 0; within_zone_pairs = 0
        cross_zone_gap = 0.0; within_zone_gap = 0.0     # dist-fields saved by each bucket's remap
        for s in states:
            nstates += 1
            T = np.nonzero(s["tm"])[0]; P = np.nonzero(s["pm"])[0]
            a = len(T); bb = len(P)
            both = s["tm"] & s["pm"]; only = s["tm"] ^ s["pm"]
            den_tot += (both.sum() + only.sum()) * nf
            # slot-matched numerator over the scored field set (positional slot excluded)
            slot_num_tot += only.sum() * nf
            if both.any():
                slot_num_tot += (s["pcat"][both][:, cat_cols] != s["tcat"][both][:, cat_cols]).sum()
                slot_num_tot += (s["pnum"][both] != s["tnum"][both]).sum()
                slot_num_tot += (s["pkw"][both] != s["tkw"][both]).any(-1).sum()
            if a == 0 and bb == 0:
                continue
            if a == 0 or bb == 0:
                content_num_tot += max(a, bb) * nf
                continue
            tcat = s["tcat"][T]; pcat = s["pcat"][P]
            tnum = s["tnum"][T]; pnum = s["pnum"][P]
            tkw = s["tkw"][T]; pkw = s["pkw"][P]
            cost = _cost(tcat, tnum, tkw, pcat, pnum, pkw, cat_cols)
            ri, ci = _assign(cost)
            matched = cost[ri, ci].sum()
            unmatched = abs(a - bb)
            content_num_tot += matched + unmatched * nf
            # identity (diagonal) cost over min(a,bb) packed rows, same cat_cols
            mmin = min(a, bb)
            ident = cost[np.arange(mmin), np.arange(mmin)].sum()
            if matched < ident - 1e-9:
                nonident += 1
                perm = -np.ones(a, int)
                for r, c in zip(ri, ci):
                    perm[r] = c
                tzone = tcat[:, ZONE_COL]
                for r in range(a):
                    c = perm[r]
                    if c == -1 or c == r:
                        continue
                    # packed target rows r and c are both in zone-major target order; compare their zones
                    same = (tzone[r] == tzone[c]) if c < a else False
                    ident_r = cost[r, r] if r < bb else FIELDS
                    saved = ident_r - cost[r, c]         # per-move field reduction (approx share)
                    if same:
                        within_zone_pairs += 1; within_zone_gap += max(0.0, saved)
                    else:
                        cross_zone_pairs += 1; cross_zone_gap += max(0.0, saved)
        res[vname] = dict(content=content_num_tot / den_tot, slot=slot_num_tot / den_tot,
                          nonident=100.0 * nonident / max(1, nstates),
                          cross=cross_zone_pairs, within=within_zone_pairs,
                          cross_gap=cross_zone_gap, within_gap=within_zone_gap)
    print(f"\n===== M2 {label}: content-matched vs slot-matched (Hungarian; slot col EXCLUDED, scored "
          f"over the same field set) =====")
    for vname in ("with-zone", "without-zone"):
        r = res[vname]
        slot = r["slot"]; gap = slot - r["content"]
        print(f"  [{vname:12s}] slot-matched = {slot:.4f}  content-matched = {r['content']:.4f}  "
              f"gap = {gap:.4f} ({100.0*gap/max(1e-9,slot):.1f}% of slot)  "
              f"non-identity states = {r['nonident']:.1f}%")
        tot = r["cross"] + r["within"]
        if tot:
            print(f"      misaligned moves: cross-zone(H1a)={r['cross']} "
                  f"({100.0*r['cross']/tot:.1f}%, gap-fields {r['cross_gap']:.0f})  "
                  f"within-zone(H1b)={r['within']} "
                  f"({100.0*r['within']/tot:.1f}%, gap-fields {r['within_gap']:.0f})")
    return res


# ==================================================================================================
# M3: error vs structure. Per present target row: mismatched fields (of FIELDS); pred-absent = FIELDS.
# ==================================================================================================
def _row_errs(s):
    """Per present target row: mismatched-field count (of FIELDS). pred-absent row = FIELDS (fully wrong)."""
    T = np.nonzero(s["tm"])[0]
    errs = np.zeros(len(T)); present_pred = np.zeros(len(T), bool)
    for k, pos in enumerate(T):
        if not s["pm"][pos]:
            errs[k] = FIELDS; continue
        present_pred[k] = True
        e = (s["pcat"][pos] != s["tcat"][pos]).sum() + (s["pnum"][pos] != s["tnum"][pos]).sum()
        e += 1 if (s["pkw"][pos] != s["tkw"][pos]).any() else 0
        errs[k] = e
    return T, errs, present_pred


def m3_structure(states, label):
    print(f"\n===== M3 {label}: error vs structure (mean mismatched fields / row, of {FIELDS}) =====")
    # by zone
    zbuck = {z: [0.0, 0] for z in range(len(ZONES))}
    # by slot index bucket
    sedges = [0, 4, 8, 12, 16, 24, 32, 48, 64]
    sbuck = {i: [0.0, 0] for i in range(len(sedges) - 1)}
    # by rows-per-state
    redges = [0, 4, 8, 12, 16, 20, 24, 200]
    rbuck = {i: [0.0, 0] for i in range(len(redges) - 1)}
    # by distance-from-zone-boundary
    dbuck = {i: [0.0, 0] for i in [0, 1, 2, 3]}   # 0,1,2,>=3
    # H1b tie-fallthrough population: rows sharing (zone,cardIndex) with another present row of a
    # DIFFERENT content key (content cats 0..5 + decoded numerics + keyword bits).
    tie_err = [0.0, 0]; notie_err = [0.0, 0]
    tie_states = 0; nstates = 0
    for s in states:
        nstates += 1
        T, errs, _ = _row_errs(s)
        k = len(T)
        if k == 0:
            continue
        tcat = s["tcat"][T]; tnum = s["tnum"][T]
        zones = tcat[:, ZONE_COL]
        # zone bucket
        for z, e in zip(zones, errs):
            if 0 <= z < len(ZONES):
                zbuck[int(z)][0] += e; zbuck[int(z)][1] += 1
        # slot bucket (position == layout index)
        for pos, e in zip(T, errs):
            bi = np.searchsorted(sedges, pos, side="right") - 1
            bi = min(bi, len(sedges) - 2)
            sbuck[bi][0] += e; sbuck[bi][1] += 1
        # rows-per-state
        ri = np.searchsorted(redges, k, side="right") - 1
        ri = min(ri, len(redges) - 2)
        rbuck[ri][0] += errs.sum(); rbuck[ri][1] += k
        # distance from zone boundary within the zone-major present layout (positions 0..k-1 contiguous)
        pos_order = np.argsort(T)     # T is already sorted ascending, keep as is
        i = 0
        while i < k:
            j = i
            while j + 1 < k and zones[j + 1] == zones[i]:
                j += 1
            run_len = j - i + 1
            for off in range(run_len):
                dist = min(off, run_len - 1 - off)
                db = dist if dist < 3 else 3
                dbuck[db][0] += errs[i + off]; dbuck[db][1] += 1
            i = j + 1
        # H1b population: same (zone,cardIndex), differing content keys. Key = content cats (0..5,
        # excludes zone/slot) + decoded numerics + keyword bits.
        cid = tcat[:, CID_COL]
        keymat = np.concatenate([tcat[:, :ZONE_COL], tnum, s["tkw"][T].astype(np.int64)], axis=1)
        tie_flag = np.zeros(k, bool)
        groups = {}
        for r in range(k):
            groups.setdefault((int(zones[r]), int(cid[r])), []).append(r)
        for g in groups.values():
            if len(g) < 2:
                continue
            keys = [tuple(int(x) for x in keymat[r]) for r in g]
            if len(set(keys)) > 1:      # same id+zone but not all identical -> tie-fallthrough
                for r in g:
                    tie_flag[r] = True
        if tie_flag.any():
            tie_states += 1
        for r in range(k):
            (tie_err if tie_flag[r] else notie_err)[0] += errs[r]
            (tie_err if tie_flag[r] else notie_err)[1] += 1

    def _line(name, d):
        return f"    {name:16s} n={d[1]:6d} err/row={d[0]/max(1,d[1]):.3f}"
    print("  -- by zone --")
    for z in range(len(ZONES)):
        print(_line(ZONES[z], zbuck[z]))
    print("  -- by slot index --")
    for i in range(len(sedges) - 1):
        print(_line(f"[{sedges[i]:2d},{sedges[i+1]:2d})", sbuck[i]))
    print("  -- by rows-per-state --")
    for i in range(len(redges) - 1):
        print(_line(f"[{redges[i]:2d},{redges[i+1]:3d})", rbuck[i]))
    print("  -- by distance from zone boundary --")
    for d in [0, 1, 2, 3]:
        print(_line(("bnd+" + str(d)) if d < 3 else "interior>=3", dbuck[d]))
    print(f"  -- H1b tie-fallthrough (same zone+cardIndex, differing dynamics) --")
    print(f"    states with >=1 tie population : {100.0*tie_states/max(1,nstates):.1f}%")
    print(f"    tie rows      n={tie_err[1]:6d} err/row={tie_err[0]/max(1,tie_err[1]):.3f}")
    print(f"    non-tie rows  n={notie_err[1]:6d} err/row={notie_err[0]/max(1,notie_err[1]):.3f}")


# ==================================================================================================
# M4: cardIndex confusion structure (slot-matched, both-present rows).
# ==================================================================================================
def m4_cardindex(states, label):
    miss_by_id = {}          # target id -> miss count
    wrong_in_state = 0; wrong_absent = 0
    for s in states:
        both = s["tm"] & s["pm"]
        pos = np.nonzero(both)[0]
        if len(pos) == 0:
            continue
        tid = s["tcat"][pos, CID_COL]; pid = s["pcat"][pos, CID_COL]
        present_ids = set(int(x) for x in s["tcat"][np.nonzero(s["tm"])[0], CID_COL])
        for t, p in zip(tid, pid):
            if t == p:
                continue
            miss_by_id[int(t)] = miss_by_id.get(int(t), 0) + 1
            if int(p) in present_ids:
                wrong_in_state += 1
            else:
                wrong_absent += 1
    tot = wrong_in_state + wrong_absent
    print(f"\n===== M4 {label}: cardIndex confusion (both-present rows) =====")
    print(f"  total cardIndex mismatches = {tot}")
    if tot:
        print(f"  wrong id present ELSEWHERE in state (copy/slot confusion) = {wrong_in_state} "
              f"({100.0*wrong_in_state/tot:.1f}%)")
        print(f"  wrong id ABSENT from state (true hallucination)          = {wrong_absent} "
              f"({100.0*wrong_absent/tot:.1f}%)")
    print(f"  top-20 most-missed cardIndex:")
    for cid, cnt in sorted(miss_by_id.items(), key=lambda kv: -kv[1])[:20]:
        nm = _CARD_CAT.id_of(cid) or "(none/id0)"
        print(f"    id={cid:4d} miss={cnt:5d}  {nm}")


# ==================================================================================================
# M5: numeric residual structure for the worst numeric columns.
# ==================================================================================================
def m5_numeric(states, label, num_input, worst_cols):
    print(f"\n===== M5 {label}: numeric residual structure (num_input={num_input}) =====")
    for j in worst_cols:
        ae_hist = [0, 0, 0]            # exact, off-by-1, off-by->=2  (on NONZERO targets)
        mag = {}                       # magnitude bucket -> [errs, n]
        buckets = [(0, 0), (1, 3), (4, 10), (11, 30), (31, 80), (81, 9999)]
        for s in states:
            both = s["tm"] & s["pm"]
            pos = np.nonzero(both)[0]
            if len(pos) == 0:
                continue
            tv = s["tnum"][pos, j]; pv = s["pnum"][pos, j]
            ae = np.abs(pv - tv)
            nz = tv != 0
            ae_hist[0] += int((ae[nz] == 0).sum())
            ae_hist[1] += int((ae[nz] == 1).sum())
            ae_hist[2] += int((ae[nz] >= 2).sum())
            for v, e in zip(tv, ae):
                for bi, (lo, hi) in enumerate(buckets):
                    if lo <= v <= hi:
                        d = mag.setdefault(bi, [0, 0]); d[0] += int(e > 0); d[1] += 1
                        break
        tot_nz = sum(ae_hist)
        print(f"  -- {NUM_NAMES[j]} --")
        if tot_nz:
            print(f"    nonzero targets n={tot_nz}: exact={ae_hist[0]/tot_nz:.3f} "
                  f"off1={ae_hist[1]/tot_nz:.3f} off>=2={ae_hist[2]/tot_nz:.3f}")
        for bi, (lo, hi) in enumerate(buckets):
            if bi in mag and mag[bi][1]:
                d = mag[bi]
                print(f"    val[{lo:>3d},{hi if hi<9999 else 999:>3d}] n={d[1]:7d} miss={d[0]/d[1]:.4f}")


def main():
    print(f"scipy={HAVE_SCIPY}   ckpt={CKPT}")
    model, meta = MF.load_checkpoint(CKPT, device="cpu")
    model.eval()
    num_input = model.num_input
    print(f"step={meta['step']} num_input={num_input} num_head={meta['config']['num_head']} "
          f"num_decode={meta['config']['num_decode']} best_state_dist={meta.get('best_state_dist'):.4f}")

    for label, loader in (("REAL", lambda: D.load_fixed_sample(CORPUS, "val", VAL_STATES)),
                          ("COVERAGE", lambda: SY.coverage_val_sample(["cards"], VAL_STATES,
                                                                      SY.COVERAGE_VAL_SEED))):
        print("\n" + "#" * 96 + f"\n########## {label} ##########")
        states, auth = load_states(model, loader)
        m1 = m1_columns(states, label, auth)
        m2_content_match(states, label)
        m3_structure(states, label)
        m4_cardindex(states, label)
        # worst 4 numeric columns by mismatch count
        worst = list(np.argsort(-m1["num_mis"])[:4])
        m5_numeric(states, label, num_input, worst)


if __name__ == "__main__":
    main()
