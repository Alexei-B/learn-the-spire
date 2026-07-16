"""Corpus streaming for the encoder/decoder trainer (roadmap 3.1).

Each corpus record carries a ``state`` and a ``nextState`` (contract 4); the world model is a per-state
autoencoder, so we mine BOTH (~2M states over the 1M-record corpus). Every state is tagged with its
fight's ``act`` (``scenarioMeta.act``) so the report card can break metrics down by act.

* :func:`iter_states` — stream ``(state, act)`` from a split, both state and nextState of each record.
* :func:`shuffle_stream` — a fixed-size shuffle buffer over an (infinitely re-iterated) source, so GPU
  batches are decorrelated without loading the corpus into RAM.
* :func:`train_batches` — infinite stream of ``(batch_tensors, acts)`` for the train loop.
* :func:`load_fixed_sample` — a deterministic first-N sample of a split, tokenized once and cached to an
  ``.npz`` (the fixed val set the trainer re-evaluates every ``--val-every`` steps).
"""

from __future__ import annotations

import os
import queue
import random
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .. import corpus, tokens
from . import model as M


def iter_states(root: str, split: str) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """Yield ``(state, act)`` for every state and nextState under ``root``/``split``."""
    for rec in corpus.iter_records(root, split=split):
        act = (rec.get("scenarioMeta") or {}).get("act")
        for which in ("state", "nextState"):
            st = rec.get(which)
            if st:
                yield st, act


def _featurize_safe(state: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    try:
        return M.featurize(state)
    except (tokens.TokenOverflow, Exception):
        return None


def shuffle_stream(root: str, split: str, buffer_size: int, rng: random.Random,
                   loop: bool = True) -> Iterator[Tuple[Dict[str, np.ndarray], Any]]:
    """Featurized ``(arrays, act)`` from a split through a shuffle buffer; re-iterates forever if
    ``loop``. Records that fail to featurize are skipped."""
    buf: List[Tuple[Dict[str, np.ndarray], Any]] = []
    while True:
        for state, act in iter_states(root, split):
            feats = _featurize_safe(state)
            if feats is None:
                continue
            if len(buf) < buffer_size:
                buf.append((feats, act))
            else:
                j = rng.randrange(buffer_size)
                out = buf[j]
                buf[j] = (feats, act)
                yield out
        if not loop:
            break
    rng.shuffle(buf)
    for item in buf:
        yield item


def train_batches_cpu(root: str, split: str, batch_size: int, buffer_size: int,
                      rng: random.Random) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite stream of ``(stacked_numpy_batch, acts)`` — CPU-side (tokenization only), so a prefetch
    thread can produce these while the GPU trains on the previous batch."""
    stream = shuffle_stream(root, split, buffer_size, rng, loop=True)
    while True:
        feats: List[Dict[str, np.ndarray]] = []
        acts: List[Any] = []
        for _ in range(batch_size):
            f, a = next(stream)
            feats.append(f)
            acts.append(a)
        yield M.collate(feats), acts


def prefetch(gen: Iterator, depth: int = 4) -> Iterator:
    """Run ``gen`` on a background thread, buffering up to ``depth`` items — overlaps the corpus-read +
    tokenization (Python/numpy, releases the GIL heavily) with the GPU step."""
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item


def train_batches(root: str, split: str, batch_size: int, buffer_size: int, device,
                  rng: random.Random, prefetch_depth: int = 4
                  ) -> Iterator[Tuple[Dict[str, torch.Tensor], List[Any]]]:
    """Infinite stream of ``(batch_tensors_on_device, acts)`` for training, tokenized on a prefetch
    thread and moved to ``device`` in the consumer."""
    cpu = train_batches_cpu(root, split, batch_size, buffer_size, rng)
    for stacked, acts in prefetch(cpu, depth=prefetch_depth):
        yield M.to_tensors(stacked, device), acts


def load_fixed_sample(root: str, split: str, n: int, cache_path: Optional[str] = None
                      ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """Deterministic first-``n`` states of a split, tokenized once (cached to ``.npz`` if given).
    Returns stacked numpy arrays + the per-state acts."""
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        stacked = {k: data[k] for k in M.BATCH_KEYS}
        return stacked, list(data["_acts"])
    feats: List[Dict[str, np.ndarray]] = []
    acts: List[Any] = []
    for state, act in iter_states(root, split):
        f = _featurize_safe(state)
        if f is None:
            continue
        feats.append(f)
        acts.append(act)
        if len(feats) >= n:
            break
    stacked = M.collate(feats)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, _acts=np.asarray(acts, dtype=object), **stacked)
    return stacked, acts


def iter_fixed_batches(stacked: Dict[str, np.ndarray], acts: List[Any], batch_size: int, device
                       ) -> Iterator[Tuple[Dict[str, torch.Tensor], List[Any]]]:
    """Iterate a cached stacked sample in eval-sized batches (no shuffle)."""
    n = len(acts)
    for i in range(0, n, batch_size):
        sl = {k: stacked[k][i:i + batch_size] for k in M.BATCH_KEYS}
        yield M.to_tensors(sl, device), acts[i:i + batch_size]
