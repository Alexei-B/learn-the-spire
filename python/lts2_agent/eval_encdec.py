"""Full-split report card for a trained world-model encoder/decoder (roadmap 3.1 CP4 artifact).

Runs a checkpoint over an ENTIRE corpus split (val by default) and prints the per-field reconstruction
report card — overall and broken down by act — the same metrics the trainer streams to the dashboard
(so the printed artifact and the live curves reference one contract). Optionally emits ``--json``.

Run::

    python -m lts2_agent.eval_encdec --ckpt checkpoints/wm_encdec.pt --split val
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import torch

from .wm import data as D
from .wm import model as M
from .wm import report


def evaluate(ckpt: str, corpus_root: str, split: str, batch_size: int, device, limit: int = 0):
    model, meta = M.load_checkpoint(ckpt, device)
    model.eval()
    accum: Dict[str, Any] = {}
    n = 0
    t0 = time.perf_counter()
    # Stream the whole split (no shuffle) in eval batches.
    feats, acts = [], []
    for state, act in D.iter_states(corpus_root, split):
        f = D._featurize_safe(state)
        if f is None:
            continue
        feats.append(f)
        acts.append(act)
        if len(feats) >= batch_size:
            batch = M.to_tensors(M.collate(feats), device)
            with torch.no_grad():
                _z, out = model(batch)
            report.merge_pairs(accum, report.report_pairs(batch, out), acts)
            n += len(acts)
            feats, acts = [], []
            if limit and n >= limit:
                break
    if feats:
        batch = M.to_tensors(M.collate(feats), device)
        with torch.no_grad():
            _z, out = model(batch)
        report.merge_pairs(accum, report.report_pairs(batch, out), acts)
        n += len(acts)
    overall, by_act = report.finalize(accum)
    dt = time.perf_counter() - t0
    return overall, by_act, n, meta, dt


def main() -> int:
    ap = argparse.ArgumentParser(description="World-model encoder/decoder full-split report card")
    ap.add_argument("--ckpt", default="checkpoints/wm_encdec.pt")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="cap states scanned (0 = whole split)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    overall, by_act, n, meta, dt = evaluate(args.ckpt, args.corpus, args.split, args.batch, device,
                                            args.limit)
    if args.json:
        print(json.dumps({"ckpt": args.ckpt, "split": args.split, "n_states": n,
                          "step": meta.get("step"), "overall": overall, "by_act": by_act}, indent=2))
    else:
        header = f"ckpt={args.ckpt} step={meta.get('step')} split={args.split} ({dt:.1f}s)"
        print(report.format_report(overall, by_act, n, header=header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
