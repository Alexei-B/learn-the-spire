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
from . import cache as C
from . import model as M
from . import spec as S
from .experts import EXPERT_TYPES


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


# --------------------------------------------------------------------------------------------------
# Cache-reading path (roadmap 3.1 speedup) — reads pre-tokenized ``.npz`` shards instead of tokenizing
# on the fly. Shuffling: shard-order shuffle + per-shard permutation, re-seeded every epoch from ``rng``
# so the whole stream is deterministic. See :mod:`lts2_agent.wm.cache`.
# --------------------------------------------------------------------------------------------------

def cache_batches_cpu(cache_dir: str, split: str, batch_size: int, rng: random.Random
                      ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite stream of ``(stacked_numpy_batch, acts)`` read from the pre-tokenized cache. Each epoch
    shuffles the shard order and permutes states within each shard; batches spanning a shard boundary
    carry the sub-batch remainder into the next shard (and across epochs)."""
    shards = C.shard_files(cache_dir, split)
    if not shards:
        raise RuntimeError(f"no cache shards for split {split!r} under {cache_dir!r}")
    carry: Optional[Tuple[Dict[str, np.ndarray], List[Any]]] = None
    while True:
        order = list(shards)
        rng.shuffle(order)
        for path in order:
            stacked, acts = C.load_shard(path)
            n = len(acts)
            perm = np.random.default_rng(rng.getrandbits(64)).permutation(n)
            stacked = {k: stacked[k][perm] for k in M.BATCH_KEYS}
            acts = [acts[i] for i in perm]
            if carry is not None:
                c_arr, c_acts = carry
                stacked = {k: np.concatenate([c_arr[k], stacked[k]]) for k in M.BATCH_KEYS}
                acts = c_acts + acts
                carry = None
                n = len(acts)
            i = 0
            while i + batch_size <= n:
                sl = {k: stacked[k][i:i + batch_size] for k in M.BATCH_KEYS}
                yield sl, acts[i:i + batch_size]
                i += batch_size
            if i < n:
                carry = ({k: stacked[k][i:] for k in M.BATCH_KEYS}, acts[i:])


# --------------------------------------------------------------------------------------------------
# Focus-present sampling (roadmap M3.5 solo-dynamics fix) — oversample states where the trained expert
# actually has tokens. A sparse category (orbs present in ~16% of states) makes a solo batch ~84% empty,
# so most gradient is spent on the presence head predicting "absent"; the id/numeric heads see almost no
# signal. This second stream draws a fraction R of each batch from PRESENT states (>=1 present token for
# the trained expert(s)) and 1-R from EMPTY states (kept so the presence head stays calibrated).
# --------------------------------------------------------------------------------------------------

def expert_present_mask(stacked: Dict[str, np.ndarray], experts: List[str]) -> np.ndarray:
    """Bool ``[n]`` — True where ANY of ``experts`` has >=1 present token in that state (union over the
    experts' variable-length token types' presence masks). Experts with only single tokens (scalars) or
    absent mask keys contribute nothing."""
    n = int(stacked["global_idx"].shape[0])
    present = np.zeros(n, dtype=bool)
    for e in experts:
        for tn in EXPERT_TYPES.get(e, []):
            t = S.TYPE_BY_NAME[tn]
            if t.mask_key and t.mask_key in stacked:
                present |= stacked[t.mask_key].reshape(n, -1).any(axis=1)
    return present


def _empty_buf() -> Tuple[Optional[Dict[str, np.ndarray]], List[Any]]:
    return None, []


def _buf_len(buf: Optional[Dict[str, np.ndarray]]) -> int:
    return 0 if buf is None else len(buf[M.BATCH_KEYS[0]])


def _buf_append(buf: Optional[Dict[str, np.ndarray]], acts: List[Any],
                add: Dict[str, np.ndarray], add_acts: List[Any]
                ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    if buf is None:
        return {k: add[k] for k in M.BATCH_KEYS}, list(add_acts)
    return ({k: np.concatenate([buf[k], add[k]]) for k in M.BATCH_KEYS}, acts + list(add_acts))


def focus_present_batches_cpu(cache_dir: str, split: str, batch_size: int, rng: random.Random,
                              experts: List[str], frac_present: float
                              ) -> Iterator[Tuple[Dict[str, np.ndarray], List[Any]]]:
    """Infinite ``(stacked_numpy_batch, acts)`` stream targeting ``round(R*batch)`` present states +
    the remainder empty states for ``experts`` (R = ``frac_present``). Same shard-shuffle + within-shard
    permute as :func:`cache_batches_cpu`; states are routed into a present pool and an empty pool, both
    **bounded** so no pool grows unboundedly (memory + O(n) work — the naive fixed-empty-count wait
    O(n²)-starves for experts that are ~always present, e.g. creatures, which have almost no empties).

    When empties are plentiful (a sparse expert like orbs, ~13% present) each batch is exactly R present.
    When the natural present rate already exceeds R (a dense expert), empties run out and the shortfall is
    filled with present states — so a dense expert's batch is ~all-present, which is the desired no-op
    (focus-present only matters for sparse experts)."""
    shards = C.shard_files(cache_dir, split)
    if not shards:
        raise RuntimeError(f"no cache shards for split {split!r} under {cache_dir!r}")
    n_present = int(round(frac_present * batch_size))
    n_empty = batch_size - n_present
    cap_p = (n_present + batch_size) if n_present > 0 else 0    # bound each pool: a couple batches' worth
    cap_e = (n_empty + batch_size) if n_empty > 0 else 0
    pres, pres_acts = _empty_buf()
    emp, emp_acts = _empty_buf()

    def _fill(buf, buf_acts, cap, add_arr, add_acts):
        room = cap - _buf_len(buf)
        if room <= 0:
            return buf, buf_acts
        return _buf_append(buf, buf_acts, {k: add_arr[k][:room] for k in M.BATCH_KEYS},
                           list(add_acts)[:room])

    while True:
        order = list(shards)
        rng.shuffle(order)
        for path in order:
            stacked, acts = C.load_shard(path)
            n = len(acts)
            perm = np.random.default_rng(rng.getrandbits(64)).permutation(n)
            stacked = {k: stacked[k][perm] for k in M.BATCH_KEYS}
            acts = [acts[i] for i in perm]
            pm = expert_present_mask(stacked, experts)
            pres, pres_acts = _fill(pres, pres_acts, cap_p, {k: stacked[k][pm] for k in M.BATCH_KEYS},
                                    [a for a, keep in zip(acts, pm) if keep])
            emp, emp_acts = _fill(emp, emp_acts, cap_e, {k: stacked[k][~pm] for k in M.BATCH_KEYS},
                                  [a for a, keep in zip(acts, ~pm) if keep])
            # Emit while a full batch (>= n_present present) can be assembled. Empties fill up to n_empty;
            # any shortfall (dense expert) is topped up with present states.
            while _buf_len(pres) >= n_present and _buf_len(pres) + _buf_len(emp) >= batch_size:
                take_e = min(n_empty, _buf_len(emp))
                take_p = batch_size - take_e
                parts, part_acts = [], []
                if take_p > 0:
                    parts.append({k: pres[k][:take_p] for k in M.BATCH_KEYS})
                    part_acts += pres_acts[:take_p]
                if take_e > 0:
                    parts.append({k: emp[k][:take_e] for k in M.BATCH_KEYS})
                    part_acts += emp_acts[:take_e]
                b_arr = {k: np.concatenate([p[k] for p in parts]) for k in M.BATCH_KEYS}
                order_in = np.random.default_rng(rng.getrandbits(64)).permutation(batch_size)
                b_arr = {k: b_arr[k][order_in] for k in M.BATCH_KEYS}
                b_acts = [part_acts[i] for i in order_in]
                yield b_arr, b_acts
                if pres is not None:
                    pres = {k: pres[k][take_p:] for k in M.BATCH_KEYS}
                    pres_acts = pres_acts[take_p:]
                if emp is not None:
                    emp = {k: emp[k][take_e:] for k in M.BATCH_KEYS}
                    emp_acts = emp_acts[take_e:]


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
                  rng: random.Random, prefetch_depth: int = 4, cache_dir: Optional[str] = None,
                  focus_experts: Optional[List[str]] = None, focus_present: float = 0.0
                  ) -> Iterator[Tuple[Dict[str, torch.Tensor], List[Any]]]:
    """Infinite stream of ``(batch_tensors_on_device, acts)`` for training.

    Uses the **pre-tokenized cache** at ``cache_dir`` when it exists and its manifest signature matches
    the current tokenizer (GPU-bound; a shard read + permute per batch). A signature mismatch raises
    loudly (rebuild). Otherwise falls back to the on-the-fly path (tokenizes on a prefetch thread) with
    a speed warning. Batches move to ``device`` in the consumer either way.

    ``focus_experts`` + ``focus_present`` (solo runs) route to :func:`focus_present_batches_cpu`, which
    oversamples states with >=1 present token for those experts (needs the cache; a warning falls back to
    the unfiltered stream otherwise)."""
    focus = bool(focus_experts) and focus_present > 0.0
    manifest = C.resolve_manifest(cache_dir)
    if manifest is not None:
        print(f"[wm.data] using pre-tokenized cache {cache_dir!r} "
              f"({manifest.get('total_states', '?')} states)", flush=True)
        if focus:
            print(f"[wm.data] focus-present: {focus_present:.2f} of each batch from states with a "
                  f"present {focus_experts} token", flush=True)
            cpu = focus_present_batches_cpu(cache_dir, split, batch_size, rng, focus_experts,
                                            focus_present)
        else:
            cpu = cache_batches_cpu(cache_dir, split, batch_size, rng)
    else:
        if focus:
            print("[wm.data] WARNING: --focus-present needs a pre-tokenized cache; ignoring (unfiltered "
                  "on-the-fly stream).", flush=True)
        if cache_dir:
            print(f"[wm.data] WARNING: no pre-tokenized cache at {cache_dir!r}; tokenizing on the fly "
                  f"(CPU-bound, slower). Build one: "
                  f"python -m lts2_agent.wm.cache build --corpus {root} --out {cache_dir}", flush=True)
        cpu = train_batches_cpu(root, split, batch_size, buffer_size, rng)
    for stacked, acts in prefetch(cpu, depth=prefetch_depth):
        yield M.to_tensors(stacked, device), acts


def load_fixed_sample_from_cache(cache_dir: str, split: str, n: int
                                 ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """First-``n`` states of a split read from the pre-tokenized cache, in cache (== corpus) order.

    The cache preserves corpus order and skips the same featurize failures the live path does, so this
    is byte-identical to the fresh ``load_fixed_sample`` first-``n`` states — the fixed val set stays
    FIXED and drawn from the val split, just read instead of retokenized."""
    feats: List[Dict[str, np.ndarray]] = []
    acts: List[Any] = []
    for path in C.shard_files(cache_dir, split):
        stacked, sh_acts = C.load_shard(path)
        for i in range(len(sh_acts)):
            feats.append({k: stacked[k][i] for k in M.BATCH_KEYS})
            acts.append(sh_acts[i])
            if len(feats) >= n:
                return M.collate(feats), acts
    return M.collate(feats), acts


def load_fixed_sample(root: str, split: str, n: int, cache_path: Optional[str] = None,
                      cache_dir: Optional[str] = None
                      ) -> Tuple[Dict[str, np.ndarray], List[Any]]:
    """Deterministic first-``n`` states of a split, tokenized once (cached to ``.npz`` if given).
    Returns stacked numpy arrays + the per-state acts. When a valid pre-tokenized ``cache_dir`` is
    present the sample is read from it (identical states); a signature mismatch raises loudly."""
    if C.resolve_manifest(cache_dir) is not None:
        return load_fixed_sample_from_cache(cache_dir, split, n)
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
