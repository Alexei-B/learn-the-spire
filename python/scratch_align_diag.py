"""Scratch diagnostic (measurement only, no product changes): how much of each population/creature
expert's residual reconstruction distance is ROW MISALIGNMENT rather than wrong content?

Hypothesis: population rows are canonically ordered by a content key that includes volatile numerics
(tokens._card_content_key for cards; within-creature sort keys for intents/powers/stats). If the decoder
gets one numeric slightly wrong, two near-tied rows swap canonical ranks and BOTH slots score fully wrong
under the slot-by-slot dist metric even though the content is nearly right. We quantify this by comparing:

  * SLOT-MATCHED dist  -- report._state_dist, exactly as report.py / the trainer computes it.
  * CONTENT-MATCHED dist -- within each state, optimally re-pair predicted PRESENT rows to target PRESENT
    rows (scipy.optimize.linear_sum_assignment over a pairwise field-mismatch cost matrix, over the SAME
    fields the dist metric counts), then score the matched pairs. Unmatched rows (presence miscount) score
    fully wrong, same as slot-matched. Uses the SAME denominator as slot-matched so gap = misalignment.

content-matched <= slot-matched always (asserted): the slot pairing is a feasible assignment, so the
optimum can only lower the numerator.

Loads the four post-split factored checkpoints directly with MF.load_checkpoint. CPU only, py3.9.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lts2_agent import tokens
from lts2_agent.wm import spec as S, model_factored as MF, report, data as D, synth as SY
from lts2_agent.wm import model as M

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

TMP = "C:/Users/Alexe/.claude/jobs/06eda07b/tmp/"
VAL_STATES = 2000
VAL_BATCH = 256
CORPUS = "data/corpus"   # reconciles cards real endpoint 0.1259 ~= recorded 0.126

# expert name -> (checkpoint, token-type name)
EXPERTS = [
    ("cards", "cards_fin50k.pt", "card"),
    ("creature-stats", "cstats_fin25k.pt", "creature"),
    ("creature-powers", "cpowers_fin25k.pt", "power"),
    ("creature-intents", "cintents_25k.pt", "intent"),
]

# recorded run endpoints (real / coverage) for reconciliation
RECORDED = {
    "cards": (0.126, 0.189),
    "creature-stats": (0.0014, 0.0186),
    "creature-powers": (0.0467, 0.0344),
    "creature-intents": (0.0149, 0.0050),
}


def _symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


def _greedy_assignment(cost):
    """Fallback if scipy is missing: greedy nearest-match over the [a,b] cost matrix. Returns
    (row_ind, col_ind) of size min(a,b)."""
    a, b = cost.shape
    order = np.argsort(cost, axis=None)
    used_r = np.zeros(a, bool)
    used_c = np.zeros(b, bool)
    ri = []
    ci = []
    for flat in order:
        r, c = divmod(int(flat), b)
        if used_r[r] or used_c[c]:
            continue
        used_r[r] = used_c[c] = True
        ri.append(r)
        ci.append(c)
        if len(ri) == min(a, b):
            break
    o = np.argsort(ri)
    return np.array(ri)[o], np.array(ci)[o]


def _assign(cost):
    if HAVE_SCIPY:
        return linear_sum_assignment(cost)
    return _greedy_assignment(cost)


def _present_fields(pred, tgt, ts):
    """Return per-state present-row field arrays for a single token-type spec ``ts``:
    (tcat, tnum_dec, tkw, pcat, pnum_dec, pkw, fields). Numerics decoded exactly as the dist metric
    (round(symexp)); kw thresholded at 0.5. tkw/pkw are None when the type has no keyword block."""
    ncat = len(ts.cat_cols)
    nnum = ts.num_width
    fields = ncat + nnum + (1 if ts.has_kw else 0)
    tm = tgt[ts.mask_key].astype(bool)
    pm = pred[ts.mask_key].astype(bool)
    T = np.nonzero(tm)[0]
    P = np.nonzero(pm)[0]
    tcat = np.atleast_2d(tgt[ts.idx_key])[T] if ts.idx_key else np.zeros((len(T), 0), int)
    pcat = np.atleast_2d(pred[ts.idx_key])[P] if ts.idx_key else np.zeros((len(P), 0), int)
    if ts.num_key:
        tnum = np.round(_symexp(np.atleast_2d(tgt[ts.num_key])[T].astype(np.float64)))
        pnum = np.round(_symexp(np.atleast_2d(pred[ts.num_key])[P].astype(np.float64)))
    else:
        tnum = np.zeros((len(T), 0)); pnum = np.zeros((len(P), 0))
    tkw = pkw = None
    if ts.has_kw:
        tkw = (np.atleast_2d(tgt["card_kw"])[T] >= 0.5)
        pkw = (np.atleast_2d(pred["card_kw"])[P] >= 0.5)
    return tcat, tnum, tkw, pcat, pnum, pkw, fields, T, P


def _cost_matrix(tcat, tnum, tkw, pcat, pnum, pkw):
    """[a,b] field-mismatch cost = (#cat cols differ) + (#num cols differ) + (1 if any kw bit differs)."""
    a = tcat.shape[0]
    b = pcat.shape[0]
    cost = np.zeros((a, b), np.float64)
    if tcat.shape[1]:
        cost += (tcat[:, None, :] != pcat[None, :, :]).sum(-1)
    if tnum.shape[1]:
        cost += (tnum[:, None, :] != pnum[None, :, :]).sum(-1)
    if tkw is not None:
        cost += (tkw[:, None, :] != pkw[None, :, :]).any(-1)
    return cost


def analyze(ename, ckpt, tname, model):
    ts = S.TYPE_BY_NAME[tname]
    typespecs = [ts]
    results = {}
    for split, loader in (("real", lambda: D.load_fixed_sample(CORPUS, "val", VAL_STATES)),
                          ("cov", lambda: SY.coverage_val_sample([ename], VAL_STATES, SY.COVERAGE_VAL_SEED))):
        stacked, acts = loader()
        n = len(acts)
        slot_num_tot = 0.0
        den_tot = 0.0
        content_num_tot = 0.0
        auth_num = []
        auth_den = []
        nonident_states = 0
        n_states = 0
        # cards swap analysis
        swap_rows_per_misaligned = []
        swap_field_hist = {}   # field-name -> count of inverted pairs whose first key diff is here
        swap_field_kind = {}   # field-name -> 'categorical'|'numeric'|'keyword'
        for i in range(0, n, VAL_BATCH):
            sl = {k: stacked[k][i:i + VAL_BATCH] for k in M.BATCH_KEYS}
            batch = M.to_tensors(sl, torch.device("cpu"))
            with torch.no_grad():
                _, outputs = model.forward(batch, active_experts=[ename])
            # authoritative slot-matched (trainer metric)
            pairs = report.report_pairs_experts_only(batch, outputs, [ename])
            an, ad = pairs["expert_dist::" + ename]
            auth_num.append(an); auth_den.append(ad)
            B = batch["global_idx"].shape[0]
            pred = report._reconstruct_types(outputs, typespecs, B)
            tgt = report._target_arrays(batch)
            for b in range(B):
                n_states += 1
                sn, sd = report._state_dist(pred[b], tgt[b], types=typespecs)
                slot_num_tot += sn
                den_tot += sd
                (tcat, tnum, tkw, pcat, pnum, pkw, fields, T, P) = _present_fields(pred[b], tgt[b], ts)
                a = len(T); bb = len(P)
                if a == 0 and bb == 0:
                    continue
                if a == 0 or bb == 0:
                    content_num_tot += max(a, bb) * fields
                    continue
                cost = _cost_matrix(tcat, tnum, tkw, pcat, pnum, pkw)
                ri, ci = _assign(cost)
                matched_cost = cost[ri, ci].sum()
                unmatched = abs(a - bb)
                content_num_tot += matched_cost + unmatched * fields
                # identity (packed diagonal) cost over the matched min(a,bb) rows
                mmin = min(a, bb)
                identity_cost = cost[np.arange(mmin), np.arange(mmin)].sum()
                if matched_cost < identity_cost - 1e-9:
                    nonident_states += 1
                    if ename == "cards":
                        # which rows moved & which key field decided each inverted pair
                        _cards_swap(tgt[b], ts, ri, ci, a, swap_rows_per_misaligned,
                                    swap_field_hist, swap_field_kind)
        slot = slot_num_tot / den_tot
        content = content_num_tot / den_tot
        auth = float(np.concatenate(auth_num).sum() / np.concatenate(auth_den).sum())
        assert content <= slot + 1e-9, f"content {content} > slot {slot} for {ename}/{split}"
        results[split] = dict(slot=slot, content=content, gap=slot - content, auth=auth,
                              pct_nonident=100.0 * nonident_states / max(1, n_states),
                              n_states=n_states,
                              swap_rows=(float(np.mean(swap_rows_per_misaligned))
                                         if swap_rows_per_misaligned else 0.0),
                              swap_field_hist=swap_field_hist, swap_field_kind=swap_field_kind)
    return results


def _cards_swap(tgt_b, ts, ri, ci, a, swap_rows_per_misaligned, swap_field_hist, swap_field_kind):
    """For a misaligned CARDS state: count moved target rows and, for every inverted pair of target
    rows (their assigned pred order is flipped vs canonical target order), record the tuple position
    where the two target rows' CONTENT KEYS first differ."""
    # perm: target packed row r -> assigned pred packed col. ri sorted ascending.
    perm = -np.ones(a, int)
    for r, c in zip(ri, ci):
        perm[r] = c
    moved = int(np.sum([1 for r in range(a) if perm[r] != -1 and perm[r] != r]))
    swap_rows_per_misaligned.append(moved)
    # target present rows -> content-key integer columns (authoritative canonical key)
    tm = tgt_b[ts.mask_key].astype(bool)
    T = np.nonzero(tm)[0]
    ci_arr = np.atleast_2d(tgt_b["card_idx"])[T]
    cn_arr = np.atleast_2d(tgt_b["card_num"])[T]
    ckw_arr = np.atleast_2d(tgt_b["card_kw"])[T].astype(np.float32)
    keycols = SY._card_content_key_columns(ci_arr, cn_arr, ckw_arr)   # list of [a] int columns
    keymat = np.stack(keycols, -1)   # [a, ncols]; cols 0..5 cat, 6..19 numeric, 20..51 keyword
    ncat = len(tokens.CARD_IDX)
    nnum_key = len([1 for c in tokens.CARD_NUM if c not in tokens.ZONE_COUNT_FIELDS])  # 14
    for x in range(a):
        for y in range(x + 1, a):
            if perm[x] == -1 or perm[y] == -1:
                continue
            if perm[x] <= perm[y]:
                continue
            # inverted pair (x<y but assigned pred cols x>y): the decoder flipped their rank
            diff = np.nonzero(keymat[x] != keymat[y])[0]
            if len(diff) == 0:
                continue
            col = int(diff[0])
            if col < ncat:
                name = tokens.CARD_IDX[col]; kind = "categorical"
            elif col < ncat + nnum_key:
                name = tokens.CARD_NUM[col - ncat]; kind = "numeric"
            else:
                name = "keywords"; kind = "keyword"
            swap_field_hist[name] = swap_field_hist.get(name, 0) + 1
            swap_field_kind[name] = kind


def main():
    print(f"scipy available: {HAVE_SCIPY}"
          + ("" if HAVE_SCIPY else "  -- using GREEDY nearest-match fallback"))
    print(f"corpus={CORPUS}  val_states={VAL_STATES}\n")
    all_res = {}
    for ename, ckpt, tname in EXPERTS:
        model, meta = MF.load_checkpoint(TMP + ckpt, device="cpu")
        model.eval()
        res = analyze(ename, ckpt, tname, model)
        all_res[ename] = res
        rec = RECORDED[ename]
        print(f"[{ename}] step={meta['step']} num_input={model.num_input}  "
              f"recorded real/cov={rec[0]}/{rec[1]}  "
              f"reconcile real={res['real']['auth']:.4f} cov={res['cov']['auth']:.4f}")
    print("\n" + "=" * 92)
    print(f"{'expert':17s} {'split':5s} {'slot':>8s} {'content':>8s} {'gap':>8s} "
          f"{'gap%ofslot':>10s} {'%non-ident':>11s}")
    print("-" * 92)
    for ename, _, _ in EXPERTS:
        for split in ("real", "cov"):
            r = all_res[ename][split]
            share = 100.0 * r["gap"] / r["slot"] if r["slot"] else 0.0
            print(f"{ename:17s} {split:5s} {r['slot']:8.4f} {r['content']:8.4f} {r['gap']:8.4f} "
                  f"{share:9.1f}% {r['pct_nonident']:10.1f}%")
    print("=" * 92)
    # cards swap-field breakdown
    for split in ("real", "cov"):
        r = all_res["cards"][split]
        print(f"\n--- cards / {split}: swapped-pair content-key analysis "
              f"(misaligned states) ---")
        print(f"  mean # swapped (moved) rows per misaligned state = {r['swap_rows']:.2f}")
        hist = r["swap_field_hist"]
        tot = sum(hist.values())
        if tot == 0:
            print("  (no inverted pairs)")
            continue
        print(f"  inverted pairs = {tot}; first-differing content-key field:")
        for name, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"    {name:14s} {r['swap_field_kind'][name]:11s} {cnt:6d}  {100.0*cnt/tot:5.1f}%")


if __name__ == "__main__":
    main()
