"""Run-comparison report for the world-model experiment series (5k-step probes and long runs).

For every metrics run dir matching the given labels/globs, this computes the three numbers the
probe protocol ranks approaches by:

* **power factor** ``b`` — the exponent of the power-law fit ``state_dist = a * step^-b`` over the
  run's ``eval.state_dist`` series (warmup excluded). Higher = converges faster per step. The fit
  quality (R^2) is reported so a bad fit can't masquerade as a good exponent.
* **p95 states/s** — the 95th percentile of the per-window ``train.states_per_s`` samples, i.e.
  the run's *uncontended* speed. This box shares its GPU with a desktop, so the mean/median are
  dragged by contention dips; the top-5% timing is what an overnight run actually sustains.
* **d @ budget** — the composite "how good a use of compute": the fitted ``state_dist`` reached
  after a fixed wall-clock budget at the run's own p95 speed
  (``steps = p95 * 3600 * hours / batch``, then ``a * steps^-b``). Reported for 1 h and 8 h.
  Ranking by ``d @ 8 h`` picks the *most effective* approach; ``b`` alone picks the most
  step-efficient; ``p95`` alone the fastest.

Usage::

    python -m lts2_agent.wm.compare                       # all wm-encdec runs
    python -m lts2_agent.wm.compare --match probe-        # only the probe series
    python -m lts2_agent.wm.compare --budget 8 --json

Fits use steps >= --fit-min (default 1000, past LR warmup). Runs without enough ``state_dist``
points are listed but unranked. Cross-tokenizer-version comparisons: ``state_dist`` units change
with the tokenizer's field universe, so compare *within* a version, or divide through by that
version's action footprint (``eval.action_snr``) — the report prints each run's tokenizer
signature so accidental cross-version ranking is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


def _load_run(run_dir: str) -> Optional[Dict[str, Any]]:
    events = os.path.join(run_dir, "events.jsonl")
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(events):
        return None
    manifest: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except ValueError:
            pass
    dist: List = []
    sps: List[float] = []
    for line in open(events, encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        n = e.get("name")
        if n == "eval.state_dist" and "tags" not in e:
            dist.append((e["step"], e["value"]))
        elif n == "train.states_per_s":
            sps.append(float(e["value"]))
    dist.sort()
    return {
        "id": os.path.basename(run_dir),
        "label": manifest.get("label") or os.path.basename(run_dir),
        "kind": manifest.get("kind"),
        "signature": (manifest.get("config") or {}).get("tokenizer_signature")
                     or manifest.get("catalogSignature"),
        "batch": int((manifest.get("config") or {}).get("batch") or 384),
        "dist": dist,
        "sps": sps,
    }


def _power_fit(dist: List, fit_min: int):
    """Fit state_dist = a * step^-b for steps >= fit_min; returns (a, b, r2, n) or None."""
    pts = [(s, v) for s, v in dist if s >= fit_min and v > 0]
    if len(pts) < 4:
        return None
    s = np.array([p[0] for p in pts], float)
    d = np.array([p[1] for p in pts], float)
    A = np.vstack([np.ones_like(s), np.log(s)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(d), rcond=None)
    a, b = float(np.exp(coef[0])), float(-coef[1])
    pred = a * s ** (-b)
    ss_res = float(np.sum((d - pred) ** 2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2, len(pts)


def analyze(run_dirs: List[str], fit_min: int, budget_hours: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        run = _load_run(rd)
        if run is None:
            continue
        row: Dict[str, Any] = {
            "label": run["label"], "id": run["id"], "signature": run["signature"], "kind": run["kind"],
            "batch": run["batch"],
            "points": len(run["dist"]),
            "last_step": run["dist"][-1][0] if run["dist"] else 0,
            "last_dist": run["dist"][-1][1] if run["dist"] else None,
            "p95_sps": float(np.percentile(run["sps"], 95)) if run["sps"] else None,
            "median_sps": float(np.median(run["sps"])) if run["sps"] else None,
        }
        fit = _power_fit(run["dist"], fit_min)
        if fit and row["p95_sps"]:
            a, b, r2, n = fit
            row.update({"a": a, "power_b": b, "fit_r2": r2, "fit_points": n})
            for h in (1.0, budget_hours):
                steps = row["p95_sps"] * 3600.0 * h / row["batch"]
                row[f"d_at_{h:g}h"] = a * steps ** (-b)
        rows.append(row)
    ranked = [r for r in rows if r.get(f"d_at_{budget_hours:g}h") is not None]
    ranked.sort(key=lambda r: r[f"d_at_{budget_hours:g}h"])
    unranked = [r for r in rows if r not in ranked]
    return ranked + unranked


def render(rows: List[Dict[str, Any]], budget_hours: float) -> str:
    key = f"d_at_{budget_hours:g}h"
    lines = ["=" * 100,
             f"{'label':<26}{'power b':>8}{'R^2':>7}{'p95 sps':>9}{'last d':>9}"
             f"{'d@1h':>9}{f'd@{budget_hours:g}h':>9}  signature",
             "-" * 100]
    for r in rows:
        b = f"{r['power_b']:.3f}" if r.get("power_b") is not None else "-"
        r2 = f"{r['fit_r2']:.3f}" if r.get("fit_r2") is not None else "-"
        sps = f"{r['p95_sps']:.0f}" if r.get("p95_sps") else "-"
        last = f"{r['last_dist']:.4f}" if r.get("last_dist") is not None else "-"
        d1 = f"{r.get('d_at_1h'):.4f}" if r.get("d_at_1h") is not None else "-"
        db = f"{r.get(key):.4f}" if r.get(key) is not None else "-"
        lines.append(f"{r['label'][:25]:<26}{b:>8}{r2:>7}{sps:>9}{last:>9}{d1:>9}{db:>9}  "
                     f"{(r.get('signature') or '?')[:28]}")
    lines.append("=" * 100)
    lines.append(f"ranked by d@{budget_hours:g}h (fitted state_dist after {budget_hours:g}h at the "
                 f"run's own p95 speed); power b = step-efficiency; p95 sps = uncontended speed.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare world-model runs: power factor, p95 speed, d@budget")
    ap.add_argument("--run-dir", default="checkpoints/runs")
    ap.add_argument("--match", default="", help="extra substring filter on run dir names")
    ap.add_argument("--fit-min", type=int, default=1000, help="fit only steps >= this (skip warmup)")
    ap.add_argument("--budget", type=float, default=8.0, help="wall-clock budget hours for d@budget")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.run_dir, "*"))
                      if os.path.isdir(d) and args.match in os.path.basename(d))
    rows = analyze(run_dirs, args.fit_min, args.budget)
    # Only world-model runs rank here (PPO/collector runs have no state_dist and just add noise).
    rows = [r for r in rows if (r.get("kind") or "").startswith("wm-") or r.get("points")]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render(rows, args.budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
