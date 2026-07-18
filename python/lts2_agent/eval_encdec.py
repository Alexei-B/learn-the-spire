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
import os
import time
from typing import Any, Dict

import torch

from .wm import data as D
from .wm import model as M
from .wm import model_factored as MF
from .wm import report


def _load_any(ckpt: str, device):
    """Load a world-model checkpoint transparently: a factored checkpoint (incl. a composed one from
    ``wm.compose``) via the factored loader, a monolith checkpoint via the mono loader. Returns
    ``(model, meta, factored)``."""
    meta_path = ckpt + ".meta.json"
    arch = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            arch = json.load(f).get("arch")
    if arch == "factored":
        model, meta = MF.load_checkpoint(ckpt, device)
        return model, meta, True
    model, meta = M.load_checkpoint(ckpt, device)
    return model, meta, False


def evaluate(ckpt: str, corpus_root: str, split: str, batch_size: int, device, limit: int = 0,
             cache_dir: str = ""):
    model, meta, factored = _load_any(ckpt, device)
    model.eval()

    def _pairs(batch, out):
        return report.report_pairs(batch, out, experts=factored)

    accum: Dict[str, Any] = {}
    n = 0
    t0 = time.perf_counter()

    def _emit(feats, acts):
        batch = M.to_tensors(M.collate(feats), device)
        with torch.no_grad():
            _z, out = model(batch)
        report.merge_pairs(accum, _pairs(batch, out), acts)

    # Read pre-tokenized shards when a cache is given (fast, CPU-friendly); else stream + tokenize.
    if cache_dir:
        stacked, acts = D.load_fixed_sample_from_cache(cache_dir, split, limit or 10 ** 9)
        for batch, b_acts in D.iter_fixed_batches(stacked, acts, batch_size, device):
            with torch.no_grad():
                _z, out = model(batch)
            report.merge_pairs(accum, _pairs(batch, out), b_acts)
            n += len(b_acts)
    else:
        feats, acts = [], []
        for state, act in D.iter_states(corpus_root, split):
            f = D._featurize_safe(state)
            if f is None:
                continue
            feats.append(f)
            acts.append(act)
            if len(feats) >= batch_size:
                _emit(feats, acts)
                n += len(acts)
                feats, acts = [], []
                if limit and n >= limit:
                    break
        if feats:
            _emit(feats, acts)
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
    ap.add_argument("--cache", default="", help="pre-tokenized cache dir to read the split from (fast); "
                                                "empty = stream + tokenize the corpus")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    overall, by_act, n, meta, dt = evaluate(args.ckpt, args.corpus, args.split, args.batch, device,
                                            args.limit, cache_dir=args.cache)
    if args.json:
        print(json.dumps({"ckpt": args.ckpt, "split": args.split, "n_states": n,
                          "step": meta.get("step"), "overall": overall, "by_act": by_act}, indent=2))
    else:
        header = f"ckpt={args.ckpt} step={meta.get('step')} split={args.split} ({dt:.1f}s)"
        print(report.format_report(overall, by_act, n, header=header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
