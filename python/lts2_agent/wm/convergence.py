"""Convergence sanity-check: is each expert learning as fast as its task entropy predicts?

The owner's heuristic: steps-to-converge should follow ``f(H) = a * H^b`` where ``H`` is the
entropy (bits/state) of the synthetic space the expert compresses. Experts sitting far ABOVE the
fitted curve are flagged LAGGING — historically that has meant a bug (ill-posed ordering, generator
canonicality, loss miscalibration), not a hard task, so the flag is a tripwire, not a verdict.

Entropies are ANALYTIC APPROXIMATIONS computed from the same constants the synth generators use
(caps, vocab sizes, NUMERIC_RANGES). They ignore small multiset/duplicate reductions — documented
per formula — which is fine for a scaling heuristic (errors of a bit or two don't move a power-law
fit materially).

Milestones are read from run metrics (coverage-val by default — the training distribution — so the
comparison is apples-to-apples across experts; real-val milestones optional). The fit uses experts
that crossed a milestone; predictions + verdicts are emitted for everyone, including uncrossed
experts (predicted crossing step vs. steps trained so far).

Caveat recorded from the first use: the canary probes ran cosine-to-50k halted early, so late-run
milestone timings are LR-confounded. For comparable milestones, run probes with the --steps big +
--halt-step trick (near-flat LR over the probe window).

Usage::

    python -m lts2_agent.wm.convergence --runs wp5-relics-synth wp5-potions-synth wp5-orbs-synth
    python -m lts2_agent.wm.convergence --runs ... --metric exact --thresholds 0.25,0.5,0.9
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import spec as S
from . import synth as SY


# --------------------------------------------------------------------------------------------------
# Analytic task entropies (bits/state) from generator constants.
# --------------------------------------------------------------------------------------------------

def _range_bits(type_name: str, col: str) -> float:
    r = S.NUMERIC_RANGES.get(type_name, {}).get(col)
    if r is None:
        return 0.0
    n = int((r.hi - r.lo) / r.resolution) + 1
    return math.log2(max(2, n))


def entropy_potions() -> float:
    """H(belt size 0..cap) + E[k] * per-slot H(empty-vs-id marginal). Ignores the multiset
    reduction (overestimates by ~1-2 bits)."""
    cap = SY.POTION_MAX_BELT
    n_ids = S.TYPE_BY_NAME["potion"].cat_cols[0][1] - 1
    e_k = cap / 2.0
    h_slot = 1.0 + 0.5 * math.log2(n_ids)      # p(empty) marginal = 0.5 under p~U(0,1)
    return math.log2(cap + 1) + e_k * h_slot


def entropy_relics() -> float:
    """H(set size) + E[k] * log2(#ids). Positional rows: slot == index carries no extra entropy;
    duplicate allowance adds a negligible fraction of a bit."""
    cap = getattr(SY, "RELIC_MAX_SET", 12)
    n_ids = S.TYPE_BY_NAME["relic"].cat_cols[0][1] - 1
    return math.log2(cap + 1) + (cap / 2.0) * math.log2(n_ids)


def entropy_orbs() -> float:
    """Reachability-shaped space: H(count) + E[k] * (log2(#real types) + mean per-type value bits
    + wildcard tail). Matches the ORB_TYPES generator (2026-07-18)."""
    cap = getattr(SY, "ORB_MAX_BELT", 12)
    types = getattr(SY, "ORB_TYPES", None)
    if not types:
        n_ids = S.TYPE_BY_NAME["orb"].cat_cols[0][1] - 1
        per_orb = math.log2(max(2, n_ids)) + _range_bits("orb", "passiveValue") + _range_bits("orb", "evokeValue")
        return math.log2(cap + 1) + (cap / 2.0) * per_orb
    per_type_bits = []
    for (plo, phi), (elo, ehi) in types.values():
        per_type_bits.append(math.log2(max(2, phi - plo + 1)) + math.log2(max(2, ehi - elo + 1)))
    w = getattr(SY, "ORB_WILDCARD_PROB", 0.05)
    per_orb = math.log2(len(types)) + sum(per_type_bits) / len(per_type_bits)
    wild = math.log2(S.TYPE_BY_NAME["orb"].cat_cols[0][1]) + _range_bits("orb", "passiveValue") + _range_bits("orb", "evokeValue")
    per_orb = (1 - w) * per_orb + w * wild
    return math.log2(cap + 1) + (cap / 2.0) * per_orb


def _hist_entropy(hist) -> float:
    p = np.asarray(hist, dtype=np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def _hist_mean(hist) -> float:
    p = np.asarray(hist, dtype=np.float64)
    return float((np.arange(len(p)) * p).sum() / p.sum())


def entropy_cards() -> float:
    """CURRENT generator (independent-uniform content): row count from the measured hist, then per row
    every categorical + dynamic numeric sampled independently, sparse keywords, game-like zone vector.
    NOTE: the real game's card fields are largely FUNCTIONS of cardIndex — this measures the GENERATOR's
    designed space, and a large value here vs. slice capacity is exactly the over-generation tripwire."""
    tspec = S.TYPE_BY_NAME["card"]
    per_row = 0.0
    for col_name, vocab in tspec.cat_cols:
        hi = vocab - 1 if col_name in ("type", "rarity", "targetType") else vocab
        per_row += math.log2(max(2, hi))
    zone_cols = set(getattr(__import__("lts2_agent.tokens", fromlist=["tokens"]), "ZONE_COUNT_FIELDS"))
    from .. import tokens as T
    for c in T.CARD_NUM:
        if c in zone_cols:
            continue
        b = _range_bits("card", c)
        per_row += b if b else 1.0                      # flag columns: 1 bit
    p_kw = 0.05                                          # generator keyword on-prob
    h_kw = -(p_kw * math.log2(p_kw) + (1 - p_kw) * math.log2(1 - p_kw))
    per_row += T.KW_BUCKETS * h_kw
    per_row += _hist_entropy(SY._CARD_NZONES_HIST) + 2.5 + 3.0   # zones chosen + per-zone counts (approx)
    return _hist_entropy(SY._CARD_ROWS_HIST) + _hist_mean(SY._CARD_ROWS_HIST) * per_row


def entropy_creatures() -> float:
    """CURRENT generator: creatures + folded powers + intents, all independent-uniform. Dominated by the
    uniform power count 0..MAX_POWERS x (id + parent + amount) term. Same over-generation caveat as
    entropy_cards()."""
    from .. import tokens as T
    cr = S.TYPE_BY_NAME["creature"]
    per_cr = math.log2(cr.cat_cols[0][1] - 1) + math.log2(cr.cat_cols[1][1])   # kind (enum) + identity
    for c in T.CREATURE_NUM:
        b = _range_bits("creature", c)
        per_cr += b if b else 1.0
    e_c = (1 + T.MAX_CREATURES) / 2.0
    pw = S.TYPE_BY_NAME["power"]
    per_pw = math.log2(pw.cat_cols[0][1]) + math.log2(e_c) + _range_bits("power", "amount")
    inn = S.TYPE_BY_NAME["intent"]
    per_in = math.log2(inn.cat_cols[0][1] - 1) + math.log2(e_c)
    for c in T.INTENT_NUM:
        b = _range_bits("intent", c)
        per_in += b if b else 1.0
    return (math.log2(T.MAX_CREATURES) + e_c * per_cr
            + math.log2(T.MAX_POWERS + 1) + (T.MAX_POWERS / 2.0) * per_pw
            + math.log2(T.MAX_INTENTS + 1) + (T.MAX_INTENTS / 2.0) * per_in)


ENTROPY_FNS = {"potions": entropy_potions, "relics": entropy_relics, "orbs": entropy_orbs,
               "cards": entropy_cards, "creatures": entropy_creatures}


# --------------------------------------------------------------------------------------------------
# Latent capacity tripwire: a SimNorm slice of width W with group g carries ~ (W/g)*log2(g) robust
# bits. An expert whose designed task entropy exceeds ~70% of that budget CANNOT reach exact coverage
# reconstruction regardless of training — flag it by arithmetic before burning GPU-hours (found live:
# orbs at H=66 vs 48-bit slice; cards/creatures generators at 2-6x their slices before reshaping).
# --------------------------------------------------------------------------------------------------

CAPACITY_WARN_FRAC = 0.7


def capacity_bits(expert: str, simnorm_group: int = 8) -> Optional[float]:
    try:
        from .model_factored import DEFAULT_SLICE_WIDTHS
    except Exception:
        return None
    w = DEFAULT_SLICE_WIDTHS.get(expert)
    return None if w is None else (w / simnorm_group) * math.log2(simnorm_group)


def capacity_report() -> List[str]:
    lines = [f"{'expert':<11}{'H bits':>9}{'cap bits':>10}{'H/cap':>7}  verdict"]
    for name, fn in ENTROPY_FNS.items():
        h = fn()
        cap = capacity_bits(name)
        if cap is None:
            continue
        ratio = h / cap
        verdict = ("OVER-CAPACITY (exact coverage impossible)" if ratio > 1.0
                   else "NEAR-CAPACITY (widen slice or shrink space)" if ratio > CAPACITY_WARN_FRAC
                   else "ok")
        lines.append(f"{name:<11}{h:>9.1f}{cap:>10.0f}{ratio:>7.2f}  {verdict}")
    return lines


def expert_entropy(name: str) -> Optional[float]:
    fn = ENTROPY_FNS.get(name)
    return fn() if fn else None


# --------------------------------------------------------------------------------------------------
# Milestones from run metrics.
# --------------------------------------------------------------------------------------------------

def _series(run_dir: str, metric: str) -> List[Tuple[int, float]]:
    pts = []
    for line in open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("name") == metric:
            pts.append((e["step"], e["value"]))
    pts.sort()
    return pts


def steps_to(pts: List[Tuple[int, float]], threshold: float, ascending: bool) -> Optional[int]:
    """First step at which the series crosses ``threshold`` (>= if ascending, <= otherwise)."""
    for s, v in pts:
        if (v >= threshold) if ascending else (v <= threshold):
            return s
    return None


def find_run(run_root: str, label: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(run_root, f"*{label}")))
    return hits[-1] if hits else None


# --------------------------------------------------------------------------------------------------
# Fit + verdicts.
# --------------------------------------------------------------------------------------------------

def fit_power(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Fit steps = a * H^b over (H, steps) points (log-log LSQ). Needs >= 2 points."""
    pts = [(h, s) for h, s in points if h and s]
    if len(pts) < 2:
        return None
    lh = np.log([p[0] for p in pts])
    ls = np.log([p[1] for p in pts])
    b, la = np.polyfit(lh, ls, 1)
    return float(np.exp(la)), float(b)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Entropy-scaled convergence sanity check (f(H)=a*H^b)")
    ap.add_argument("--run-dir", default="checkpoints/runs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run labels (suffix match), one per expert, e.g. wp5-potions-synth ...")
    ap.add_argument("--experts", nargs="+", default=None,
                    help="expert name per run (default: inferred from the label)")
    ap.add_argument("--metric", default="dist", choices=["dist", "exact"],
                    help="coverage metric family: dist (descending) or exact (ascending)")
    ap.add_argument("--thresholds", default=None,
                    help="comma list; defaults: dist -> 0.2,0.1,0.05 / exact -> 0.25,0.5,0.9")
    ap.add_argument("--lag-factor", type=float, default=2.0,
                    help="actual/predicted ratio above which an expert is flagged LAGGING")
    args = ap.parse_args(argv)

    print("-- latent capacity vs designed task entropy --")
    for ln in capacity_report():
        print(ln)
    print()

    ascending = args.metric == "exact"
    metric_name = "eval.expert_exact_cov" if ascending else "eval.expert_dist_cov"
    ths = [float(x) for x in (args.thresholds.split(",") if args.thresholds
                              else (["0.25", "0.5", "0.9"] if ascending else ["0.2", "0.1", "0.05"]))]

    rows = []
    for i, label in enumerate(args.runs):
        expert = (args.experts[i] if args.experts else
                  next((e for e in ENTROPY_FNS if e in label), None))
        d = find_run(args.run_dir, label)
        if d is None or expert is None:
            print(f"!! {label}: run or expert name not resolved; skipped")
            continue
        pts = _series(d, metric_name)
        h = expert_entropy(expert)
        row = {"label": label, "expert": expert, "H": h,
               "trained": pts[-1][0] if pts else 0, "last": pts[-1][1] if pts else None,
               "cross": {t: steps_to(pts, t, ascending) for t in ths}}
        rows.append(row)

    print(f"metric={metric_name}  thresholds={ths}")
    print(f"{'expert':<10}{'H bits':>8}{'trained':>9}{'last':>8}"
          + "".join(f"{'@' + str(t):>9}" for t in ths))
    for r in rows:
        last = "-" if r["last"] is None else "%.3f" % r["last"]
        cross_cols = "".join("%9s" % (r["cross"][t] if r["cross"][t] else "-") for t in ths)
        print("%-10s%8.1f%9d%8s%s" % (r["expert"], r["H"], r["trained"], last, cross_cols))

    print()
    for t in ths:
        pts = [(r["H"], r["cross"][t]) for r in rows if r["cross"][t]]
        fit = fit_power(pts)
        if fit is None:
            print(f"@{t}: <2 crossings, no fit yet")
            continue
        a, b = fit
        print(f"@{t}: steps ~= {a:.3g} * H^{b:.2f}   (fit on {len(pts)} experts)")
        for r in rows:
            pred = a * (r["H"] ** b)
            actual = r["cross"][t]
            if actual:
                ratio = actual / pred
                flag = "LAGGING" if ratio > args.lag_factor else ("fast" if ratio < 1 / args.lag_factor else "on-curve")
                print(f"    {r['expert']:<10} actual {actual:>7}  predicted {pred:>9.0f}  x{ratio:.2f}  {flag}")
            else:
                verdict = "LAGGING (past prediction, not crossed)" if r["trained"] > args.lag_factor * pred \
                    else f"pending (predicted ~{pred:.0f}, trained {r['trained']})"
                print(f"    {r['expert']:<10} not crossed — {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
