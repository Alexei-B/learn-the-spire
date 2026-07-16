"""Pre-tokenized corpus cache (roadmap 3.1 speedup).

The on-the-fly loader in :mod:`lts2_agent.wm.data` tokenizes every state in Python as the GPU trains,
which is CPU-bound (~400 states/s) and re-does the identical work on every run over a corpus that never
changes. This module builds a **one-time, pre-tokenized cache**: it streams every corpus record per
split, tokenizes states with a **multiprocessing pool** (the tokenizer is pure numpy, so workers scale),
and writes fixed-size compressed ``.npz`` array shards under ``<out>/<split>/`` plus a ``manifest.json``
carrying the tokenizer signature so a stale cache rejects loudly.

Dedup decision — **both states kept (no dedup)**
------------------------------------------------
Within a fight, record ``t``'s ``nextState`` equals record ``t+1``'s ``state``, so the corpus is ~2x
redundant. We nonetheless store **one cached state per (record, which in {state, nextState})** — exactly
what :func:`data.iter_states` yields — rather than deduping to unique states (each record's ``state`` +
the final ``nextState`` per fight). Reason: **training-distribution parity**. The live shuffle buffer
draws uniformly from the multiset produced by ``iter_states``; interior states therefore appear with
multiplicity 2 and fight-boundary states with multiplicity 1. Deduping would halve the relative
frequency of interior states, changing the distribution the model sees per epoch. The cache is a drop-in
for the fresh loader, so it preserves that multiset exactly (state count ~= 2*records - featurize-skips).

States that fail to featurize are **skipped** (not stored), identically to
:func:`data._featurize_safe` returning ``None`` in the live path, so the cached order matches the fresh
order position-for-position (the pool preserves input order) — which is what makes the fixed val sample
and the ``--verify`` byte-equality check exact.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .. import corpus, tokens
from . import model as M

MANIFEST_NAME = "manifest.json"
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".npz"
ACTS_KEY = "_acts"
DEFAULT_SHARD_SIZE = 4096


# ==================================================================================================
# Tokenization worker (top-level so it pickles into pool processes).
# ==================================================================================================

def _tokenize_one(item: Tuple[Dict[str, Any], Any]
                  ) -> Tuple[Optional[Dict[str, np.ndarray]], Any]:
    """Featurize one ``(state, act)`` -> ``(arrays_or_None, act)``. Mirrors ``data._featurize_safe``:
    a state that fails to tokenize returns ``None`` and is skipped by the builder."""
    state, act = item
    try:
        return M.featurize(state), act
    except Exception:
        return None, act


def _iter_state_act(root: str, split: str) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """Same source the live loader mines: ``(state, act)`` for every state AND nextState of a split."""
    for rec in corpus.iter_records(root, split=split):
        act = (rec.get("scenarioMeta") or {}).get("act")
        for which in ("state", "nextState"):
            st = rec.get(which)
            if st:
                yield st, act


# ==================================================================================================
# Shard IO.
# ==================================================================================================

def shard_files(cache_dir: str, split: str) -> List[str]:
    """Sorted list of ``.npz`` shard paths for one split (empty if the split dir is absent)."""
    d = os.path.join(cache_dir, split)
    if not os.path.isdir(d):
        return []
    names = [n for n in os.listdir(d) if n.startswith(SHARD_PREFIX) and n.endswith(SHARD_SUFFIX)]
    return [os.path.join(d, n) for n in sorted(names)]


def _write_shard(path: str, feats: List[Dict[str, np.ndarray]], acts: List[Any]) -> None:
    stacked = M.collate(feats)
    np.savez_compressed(path, _acts=np.asarray(acts, dtype=object), **stacked)


def load_shard(path: str) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """Load one shard -> ``(stacked_arrays, acts)``. ``stacked_arrays`` keyed by ``M.BATCH_KEYS``."""
    data = np.load(path, allow_pickle=True)
    stacked = {k: data[k] for k in M.BATCH_KEYS}
    return stacked, list(data[ACTS_KEY])


# ==================================================================================================
# Manifest.
# ==================================================================================================

def read_manifest(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Return the parsed manifest, or ``None`` if the cache dir / manifest is absent."""
    path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def resolve_manifest(cache_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Validate a cache dir for use by the loader. Returns the manifest when present AND its tokenizer
    signature matches the current tokenizer; returns ``None`` when the cache dir / manifest is absent
    (caller falls back to on-the-fly). Raises ``RuntimeError`` on a **signature mismatch** — a stale
    cache must reject loudly, never silently retokenize under a mismatched contract."""
    if not cache_dir:
        return None
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return None
    want = tokens.tokenizer_signature()
    got = manifest.get("tokenizer_signature")
    if got != want:
        raise RuntimeError(
            f"Pre-tokenized cache at {cache_dir!r} was built with a different tokenizer/catalog "
            f"(manifest={got!r} vs current {want!r}). Rebuild it:\n"
            f"    python -m lts2_agent.wm.cache build --corpus <corpus> --out {cache_dir}")
    return manifest


# ==================================================================================================
# Builder.
# ==================================================================================================

def build_split(root: str, split: str, out_dir: str, pool, shard_size: int,
                chunksize: int = 64) -> Dict[str, Any]:
    """Tokenize one split into ``<out_dir>/<split>/shard-NNNNN.npz`` shards (order-preserving).
    Returns a per-split stats dict for the manifest."""
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    feats_buf: List[Dict[str, np.ndarray]] = []
    acts_buf: List[Any] = []
    n_states = 0
    n_skipped = 0
    n_shards = 0

    def flush() -> None:
        nonlocal n_shards
        if not feats_buf:
            return
        path = os.path.join(split_dir, f"{SHARD_PREFIX}{n_shards:05d}{SHARD_SUFFIX}")
        _write_shard(path, feats_buf, acts_buf)
        n_shards += 1
        feats_buf.clear()
        acts_buf.clear()

    src = _iter_state_act(root, split)
    results = pool.imap(_tokenize_one, src, chunksize=chunksize) if pool is not None \
        else (_tokenize_one(x) for x in src)
    for feats, act in results:
        if feats is None:
            n_skipped += 1
            continue
        feats_buf.append(feats)
        acts_buf.append(act)
        n_states += 1
        if len(feats_buf) >= shard_size:
            flush()
    flush()
    return {"n_states": n_states, "n_skipped": n_skipped, "n_shards": n_shards,
            "shard_size": shard_size}


def build(root: str, out_dir: str, workers: int = 8, shard_size: int = DEFAULT_SHARD_SIZE,
          splits: Tuple[str, ...] = corpus.SPLITS) -> Dict[str, Any]:
    """Build the full pre-tokenized cache under ``out_dir`` and write its manifest. Reports build rate
    per split. Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    pool = mp.Pool(processes=workers) if workers and workers > 1 else None
    per_split: Dict[str, Any] = {}
    t_all = time.perf_counter()
    try:
        for split in splits:
            t0 = time.perf_counter()
            stats = build_split(root, split, out_dir, pool, shard_size)
            dt = time.perf_counter() - t0
            rate = stats["n_states"] / max(1e-6, dt)
            stats["build_seconds"] = round(dt, 1)
            stats["build_states_per_s"] = round(rate, 1)
            per_split[split] = stats
            print(f"[cache] {split:5s}: {stats['n_states']:>9,d} states  "
                  f"{stats['n_shards']:>4d} shards  {dt:7.1f}s  {rate:8.0f} states/s  "
                  f"(skipped {stats['n_skipped']})", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    total_states = sum(s["n_states"] for s in per_split.values())
    total_dt = time.perf_counter() - t_all
    manifest = {
        "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
        "corpus_root": os.path.abspath(root),
        "dedup": "both-states",
        "batch_keys": list(M.BATCH_KEYS),
        "shard_size": shard_size,
        "workers": workers,
        "total_states": total_states,
        "build_seconds": round(total_dt, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": per_split,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[cache] total {total_states:,} states in {total_dt:.1f}s "
          f"({total_states / max(1e-6, total_dt):.0f} states/s) -> {out_dir}", flush=True)
    return manifest


# ==================================================================================================
# Equivalence check — the correctness gate.
# ==================================================================================================

def verify(root: str, cache_dir: str, split: str, n_sample: int = 200,
           seed: int = 0) -> Tuple[int, int]:
    """Re-tokenize ``n_sample`` random cached states fresh from the corpus and assert byte-equality of
    every array. Returns ``(checked, total_states)``; raises ``AssertionError`` on any mismatch.

    Cached order == fresh order (the pool preserves input order and both skip the same featurize
    failures), so the k-th cached state must equal the k-th freshly-featurized state byte-for-byte."""
    manifest = read_manifest(cache_dir)
    total = int((manifest or {}).get("splits", {}).get(split, {}).get("n_states", 0))
    if total == 0:
        raise AssertionError(f"cache split {split!r} is empty; nothing to verify")
    rng = random.Random(seed)
    n_sample = min(n_sample, total)
    want_idx = sorted(rng.sample(range(total), n_sample))
    want_set = set(want_idx)
    max_idx = want_idx[-1]

    # Stream cached states (in order) and fresh-featurized states (in order) in lockstep.
    def cached_states() -> Iterator[Dict[str, np.ndarray]]:
        for path in shard_files(cache_dir, split):
            stacked, acts = load_shard(path)
            for i in range(len(acts)):
                yield {k: stacked[k][i] for k in M.BATCH_KEYS}

    fresh = _iter_state_act(root, split)
    fresh_feats = (f for f, _ in (_tokenize_one(x) for x in fresh))

    checked = 0
    cache_it = cached_states()
    idx = 0
    for cached in cache_it:
        # Advance the fresh stream to the next successfully-featurized state.
        fresh_arr = None
        while fresh_arr is None:
            fresh_arr = next(fresh_feats)
        if idx in want_set:
            for k in M.BATCH_KEYS:
                a, b = cached[k], fresh_arr[k]
                if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
                    raise AssertionError(
                        f"cache mismatch at {split}#{idx} key {k!r}: "
                        f"cached{a.shape}/{a.dtype} vs fresh{b.shape}/{b.dtype}")
            checked += 1
        idx += 1
        if idx > max_idx:
            break
    if checked != n_sample:
        raise AssertionError(f"verify only checked {checked}/{n_sample} states (cache too short?)")
    return checked, total


# ==================================================================================================
# CLI.
# ==================================================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-tokenized corpus cache builder (roadmap 3.1 speedup).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="tokenize a corpus into pre-tokenized array shards")
    b.add_argument("--corpus", default="data/corpus", help="corpus root to read")
    b.add_argument("--out", default="data/corpus_tok", help="output cache dir")
    b.add_argument("--workers", type=int, default=8, help="tokenization pool size")
    b.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="states per shard")
    b.add_argument("--splits", nargs="+", default=list(corpus.SPLITS))
    b.add_argument("--verify", action="store_true",
                   help="after building, re-tokenize a random sample fresh and assert byte-equality")
    b.add_argument("--verify-n", type=int, default=200, help="states to check in --verify")
    b.add_argument("--verify-split", default="val", help="split to sample for --verify")
    b.add_argument("--no-verify", action="store_true",
                   help="skip the automatic post-build equivalence check")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.corpus, args.out, workers=args.workers, shard_size=args.shard_size,
              splits=tuple(args.splits))
        if not args.no_verify:
            print(f"[cache] verifying {args.verify_n} random {args.verify_split} states "
                  f"against a fresh tokenize...", flush=True)
            checked, total = verify(args.corpus, args.out, args.verify_split, args.verify_n)
            print(f"[cache] VERIFY PASS: {checked}/{total} {args.verify_split} states byte-identical "
                  f"to a fresh tokenize.", flush=True)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
