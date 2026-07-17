"""Action-footprint measurement — the SNR denominator scale for the report card (roadmap 3.1).

:data:`lts2_agent.wm.report.ACTION_FOOTPRINT` is the *median fraction of token-fields a single real
action changes* (``state`` -> ``nextState``), measured in the SAME ``_state_dist`` token-field units the
decoder's reconstruction error is measured in. The report's ``action_snr`` metric divides that footprint
by the decoder's per-field distance — so "how many actions' worth of change can the decoder still tell
apart?" is a scale-free number.

The footprint is a **function of the tokenizer's field universe**, so it must be re-measured whenever the
tokenizer layout changes (e.g. v2 count-grouped tokens -> v3 factored population rows: zone leaves the
grouping key and becomes a per-zone count vector, so a PlayCard mostly shifts counts between two columns
of one row instead of moving a whole card token — changing both the numerator and denominator of every
transition's field distance). This module streams a split, tokenizes both states of each transition under
the CURRENT tokenizer, computes the per-transition footprint, and reports the median overall + per action
kind, printing the constant to paste into :data:`report.ACTION_FOOTPRINT`.

CLI::

    python -m lts2_agent.wm.footprint --corpus data/corpus --n 3000   # val split, 3000 transitions

CPU-only, corpus-streaming; no torch, no GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

from .. import corpus, tokens
from .report import ACTION_FOOTPRINT, _state_dist


def action_kind(rec: Dict[str, Any]) -> str:
    """The kind of the action taken in ``rec`` (``PlayCard`` / ``EndTurn`` / ``SelectCards`` / ...).

    ``actionTaken`` is an option index (int) or a list of card indices (a multi-select choice)."""
    at = rec.get("actionTaken")
    if isinstance(at, list):
        return "SelectCards"
    opts = rec.get("options") or []
    if isinstance(at, int) and 0 <= at < len(opts):
        return (opts[at] or {}).get("kind") or "Other"
    return "Other"


def measure(root: str, split: str = "val", n: Optional[int] = 3000
            ) -> Tuple[List[float], Dict[str, List[float]]]:
    """Stream up to ``n`` transitions of ``split`` and return ``(all_footprints, per_kind_footprints)``.

    Each footprint is ``_state_dist(tokenize(state), tokenize(nextState))`` as a fraction (num/den) — the
    share of token-fields the action changed. Transitions with no ``nextState`` (fight-ending / terminal)
    or that fail to tokenize are skipped."""
    fps: List[float] = []
    per_kind: Dict[str, List[float]] = {}
    seen = 0
    for rec in corpus.iter_records(root, split):
        st = rec.get("state")
        nx = rec.get("nextState")
        if not st or not nx:
            continue
        try:
            ta = tokens.tokenize(st, strict=False)
            tb = tokens.tokenize(nx, strict=False)
        except Exception:
            continue
        num, den = _state_dist(ta, tb)
        fp = num / den if den else 0.0
        fps.append(fp)
        per_kind.setdefault(action_kind(rec), []).append(fp)
        seen += 1
        if n and seen >= n:
            break
    return fps, per_kind


def _stats(xs: List[float]) -> Tuple[float, float, int]:
    if not xs:
        return 0.0, 0.0, 0
    return statistics.median(xs), statistics.mean(xs), len(xs)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Measure the per-action state_dist footprint (report SNR).")
    ap.add_argument("--corpus", default="data/corpus", help="corpus root to stream")
    ap.add_argument("--split", default="val", help="split to sample (default val)")
    ap.add_argument("--n", type=int, default=3000, help="transitions to measure (0 = whole split)")
    args = ap.parse_args(argv)

    fps, per_kind = measure(args.corpus, args.split, args.n or None)
    if not fps:
        print("no transitions measured (empty split or all skipped)")
        return 1

    overall_med, overall_mean, n = _stats(fps)
    print("=" * 72)
    print(f"ACTION FOOTPRINT   ({tokens.tokenizer_signature()})")
    print("=" * 72)
    print(f"split {args.split!r}   transitions {n}   root {args.corpus}")
    print()
    print(f"  {'kind':14s} {'count':>7s} {'median':>9s} {'mean':>9s}")
    for kind in sorted(per_kind, key=lambda k: -len(per_kind[k])):
        med, mean, c = _stats(per_kind[kind])
        print(f"  {kind:14s} {c:7d} {med:9.4f} {mean:9.4f}")
    print(f"  {'OVERALL':14s} {n:7d} {overall_med:9.4f} {overall_mean:9.4f}")
    print()
    print(f"current report.ACTION_FOOTPRINT = {ACTION_FOOTPRINT}")
    print(f"=> set report.ACTION_FOOTPRINT = {overall_med:.4f}  (overall median under this tokenizer)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
