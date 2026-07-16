"""Transition corpus: sharded, compressed, append-only storage of logged game transitions.

The world-model (encoder/predictor) is trained **supervised** on logged transitions. This module is
the corpus store: one record per decision point = ``(state, action, next-state)`` plus the metadata a
downstream trainer needs to tag/stratify and to split leak-free. Records are the **raw wire
observations** (lossless), so they can be replayed through any future tokenizer forever.

Layout — sharded gzip JSONL under a corpus root::

    <root>/train/<shard-id>.jsonl.gz
    <root>/val/<shard-id>.jsonl.gz
    <root>/test/<shard-id>.jsonl.gz

One JSON object per line, one line per decision. Shards are capped at ~2000 records; a new shard rolls
over when the cap is hit. A whole fight is written **atomically** (all its records or none) so a
partial/errored fight never lands in the corpus.

Split discipline (leak-proof, deterministic): the split is a pure function of the **fight seed** —
``crc32(seed) % 100`` → 0-89 train / 90-94 val / 95-99 test. All records of one fight share the seed,
so a fight lands wholly in one split; the same seed always maps to the same split, in the writer and in
any future reader. That makes train/val/test leakage structurally impossible.

Seed discipline: probe seeds (the oracle's ``PROBE-`` eval namespace) are refused, and records whose
scenario came from an **explicit** deckSpec are refused — fixed instances (closed-eval) must never
enter training data (the roadmap's anti-overfit rule). Stdlib only.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import zlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .oracle import assert_not_probe_seed

# --------------------------------------------------------------------------------------------------
# Split assignment — the leak-proof, deterministic seed -> split function.
# --------------------------------------------------------------------------------------------------

SPLITS = ("train", "val", "test")

# Bucket boundaries over crc32(seed) % 100. One function, used by the writer and any reader.
_TRAIN_MAX = 89   # 0..89  -> train (90%)
_VAL_MAX = 94     # 90..94 -> val   (5%)
#                   95..99 -> test  (5%)


def split_bucket(seed: str) -> int:
    """The stable 0-99 bucket for ``seed`` (``crc32`` of its UTF-8 bytes, mod 100)."""
    return zlib.crc32(seed.encode("utf-8")) % 100


def split_for_seed(seed: str) -> str:
    """Assign ``seed`` to ``"train"``/``"val"``/``"test"`` deterministically from its crc32 bucket.

    Pure and total: identical for the writer and every future reader, so a fight seed can never appear
    in two splits.
    """
    bucket = split_bucket(seed)
    if bucket <= _TRAIN_MAX:
        return "train"
    if bucket <= _VAL_MAX:
        return "val"
    return "test"


# --------------------------------------------------------------------------------------------------
# Record schema (roadmap contract 4).
# --------------------------------------------------------------------------------------------------

# The exact top-level keys of a corpus record, in order.
RECORD_KEYS = (
    "seed", "scenarioMeta", "t", "state", "options", "actionTaken",
    "nextState", "nextOptions", "rewardComponents", "done", "info",
)

# scenarioMeta keys (all present, possibly null).
SCENARIO_META_KEYS = (
    "deckSpec", "removedCards", "addedCards",
    "addedRelics", "addedPotions", "starterRelicState", "upgradedStarterRelic",
    "act", "room", "character", "encounter", "policy", "regime",
)


def _enemy_hp_sum(state: Optional[Dict[str, Any]]) -> int:
    """Sum of living-enemy current HP in ``state`` (0 when out of combat / no enemies)."""
    if not state:
        return 0
    combat = state.get("combat") or {}
    total = 0
    for enemy in combat.get("enemies") or []:
        hp = enemy.get("currentHp")
        if hp:
            total += hp
    return total


def _player_scalar(state: Optional[Dict[str, Any]], key: str) -> int:
    if not state:
        return 0
    players = state.get("players") or []
    if not players:
        return 0
    return players[0].get(key) or 0


def reward_components(state: Dict[str, Any], next_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Raw before/after scalars for a transition — **no reward function applied** (kept lossless).

    Player currentHp/block and the summed enemy HP, measured on the pre-step ``state`` and the post-step
    ``next_state``. A downstream trainer derives whatever reward it wants from these.
    """
    return {
        "currentHp": {"before": _player_scalar(state, "currentHp"),
                      "after": _player_scalar(next_state, "currentHp")},
        "block": {"before": _player_scalar(state, "block"),
                  "after": _player_scalar(next_state, "block")},
        "enemyHp": {"before": _enemy_hp_sum(state),
                    "after": _enemy_hp_sum(next_state)},
    }


def make_scenario_meta(
    *, deck_spec: Optional[str], removed_cards: Optional[Sequence[str]],
    added_cards: Optional[Sequence[str]], act: Any, room: Any, character: Any,
    encounter: Any, policy: str, regime: str,
    added_relics: Optional[Sequence[str]] = None, added_potions: Optional[Sequence[str]] = None,
    starter_relic_state: Optional[str] = None, upgraded_starter_relic: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the ``scenarioMeta`` block (all contract-4 keys, nulls kept explicit). The relic /
    potion / starter-relic fields mirror the realistic deckSpec's scenario metadata in ``info``
    (addedRelics / addedPotions / starterRelicState / upgradedStarterRelic); null for other kinds."""
    return {
        "deckSpec": deck_spec,
        "removedCards": list(removed_cards) if removed_cards is not None else None,
        "addedCards": list(added_cards) if added_cards is not None else None,
        "addedRelics": list(added_relics) if added_relics is not None else None,
        "addedPotions": list(added_potions) if added_potions is not None else None,
        "starterRelicState": starter_relic_state,
        "upgradedStarterRelic": upgraded_starter_relic,
        "act": act,
        "room": room,
        "character": character,
        "encounter": encounter,
        "policy": policy,
        "regime": regime,
    }


def make_record(
    *, seed: str, scenario_meta: Dict[str, Any], t: int,
    state: Dict[str, Any], options: List[Dict[str, Any]], action_taken: Any,
    next_state: Optional[Dict[str, Any]], next_options: Optional[List[Dict[str, Any]]],
    done: bool, info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one corpus record (contract 4). ``action_taken`` is an ``int`` (option index) or a
    ``list[int]`` (card indices for a multi-select choice). ``rewardComponents`` is derived raw."""
    return {
        "seed": seed,
        "scenarioMeta": scenario_meta,
        "t": int(t),
        "state": state,
        "options": options,
        "actionTaken": action_taken,
        "nextState": next_state,
        "nextOptions": next_options,
        "rewardComponents": reward_components(state, next_state),
        "done": bool(done),
        "info": info,
    }


def validate_record(rec: Dict[str, Any]) -> None:
    """Assert ``rec`` has the contract-4 shape; raise ``ValueError`` on the first violation."""
    for key in RECORD_KEYS:
        if key not in rec:
            raise ValueError(f"corpus record missing key {key!r}")
    if not isinstance(rec["seed"], str):
        raise ValueError("record 'seed' must be a string")
    meta = rec["scenarioMeta"]
    if not isinstance(meta, dict):
        raise ValueError("record 'scenarioMeta' must be an object")
    for key in SCENARIO_META_KEYS:
        if key not in meta:
            raise ValueError(f"scenarioMeta missing key {key!r}")
    if not isinstance(rec["t"], int):
        raise ValueError("record 't' must be an int")
    action = rec["actionTaken"]
    if not (isinstance(action, int) or (isinstance(action, list)
                                        and all(isinstance(i, int) for i in action))):
        raise ValueError("record 'actionTaken' must be an int or a list of ints")


class ExplicitDeckSpecError(ValueError):
    """Raised when a record from an explicit (closed-eval) deckSpec is offered to the writer."""


def assert_writable(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Guard a record before it enters the corpus: reject probe seeds and explicit deckSpecs.

    Enforces the roadmap's anti-overfit rule structurally — probe fights (``PROBE-`` namespace) and
    fixed closed-eval instances (``deckSpec == "explicit"``) can never become training data.
    """
    assert_not_probe_seed(rec["seed"])
    deck_spec = (rec.get("scenarioMeta") or {}).get("deckSpec")
    regime = (rec.get("scenarioMeta") or {}).get("regime")
    if deck_spec == "explicit" or regime == "explicit":
        raise ExplicitDeckSpecError(
            f"record seed {rec['seed']!r} came from an explicit deckSpec; collectors refuse explicit "
            "specs — fixed instances are eval-only and must never enter the training corpus."
        )
    return rec


# --------------------------------------------------------------------------------------------------
# Sharded writer.
# --------------------------------------------------------------------------------------------------


class CorpusWriter:
    """Thread-safe, append-only writer of sharded gzip-JSONL transitions under a corpus root.

    Records are routed to ``<root>/<split>/<shard-id>.jsonl.gz`` by the fight seed's split. A whole
    fight is written with :meth:`write_fight` **atomically** under a lock: either all of its records
    land (contiguously, into that fight's split) or none do, so a caller that drops an errored fight
    never leaves a partial one behind. Shards roll over at ``shard_cap`` records.
    """

    def __init__(self, root: str, run_label: str = "corpus", shard_cap: int = 2000,
                 validate: bool = True):
        self.root = root
        self.run_label = run_label
        self.shard_cap = shard_cap
        self.validate = validate
        self._lock = threading.Lock()
        # Per-split open shard state.
        self._fh: Dict[str, Any] = {}
        self._count: Dict[str, int] = {}      # records in the currently-open shard
        self._seq: Dict[str, int] = {}        # next shard index for this split
        self._total: Dict[str, int] = {s: 0 for s in SPLITS}
        for split in SPLITS:
            os.makedirs(os.path.join(root, split), exist_ok=True)
            self._seq[split] = self._next_seq(split)

    def _next_seq(self, split: str) -> int:
        """First shard index not already present for ``split`` (so re-runs never clobber shards)."""
        d = os.path.join(self.root, split)
        prefix = f"{self.run_label}-"
        existing = []
        for name in os.listdir(d):
            if name.startswith(prefix) and name.endswith(".jsonl.gz"):
                stem = name[len(prefix):-len(".jsonl.gz")]
                if stem.isdigit():
                    existing.append(int(stem))
        return (max(existing) + 1) if existing else 0

    def _shard_path(self, split: str, seq: int) -> str:
        return os.path.join(self.root, split, f"{self.run_label}-{seq:05d}.jsonl.gz")

    def _open_shard(self, split: str) -> None:
        seq = self._seq[split]
        self._fh[split] = gzip.open(self._shard_path(split, seq), "at", encoding="utf-8")
        self._count[split] = 0

    def _roll_if_needed(self, split: str) -> None:
        if split not in self._fh:
            self._open_shard(split)
        elif self._count[split] >= self.shard_cap:
            self._fh[split].close()
            self._seq[split] += 1
            self._open_shard(split)

    def write_fight(self, records: Sequence[Dict[str, Any]]) -> int:
        """Write all of one fight's records atomically. Returns the number written (0 for an empty
        fight). Raises before writing anything if any record fails validation or the seed/deckSpec
        guard — so a rejected fight leaves the corpus untouched."""
        records = list(records)
        if not records:
            return 0
        for rec in records:
            if self.validate:
                validate_record(rec)
            assert_writable(rec)
        split = split_for_seed(records[0]["seed"])
        with self._lock:
            for rec in records:
                self._roll_if_needed(split)
                self._fh[split].write(json.dumps(rec, separators=(",", ":")) + "\n")
                self._count[split] += 1
                self._total[split] += 1
            self._fh[split].flush()
        return len(records)

    @property
    def totals(self) -> Dict[str, int]:
        """Records written per split so far (a snapshot copy)."""
        with self._lock:
            return dict(self._total)

    def close(self) -> None:
        with self._lock:
            for fh in self._fh.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._fh.clear()

    def __enter__(self) -> "CorpusWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------------------------------
# Reader.
# --------------------------------------------------------------------------------------------------


def shard_paths(root: str, split: Optional[str] = None) -> List[str]:
    """Sorted list of shard files under ``root`` (optionally restricted to one ``split``)."""
    splits = [split] if split else list(SPLITS)
    paths: List[str] = []
    for s in splits:
        d = os.path.join(root, s)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".jsonl.gz"):
                paths.append(os.path.join(d, name))
    return sorted(paths)


def iter_shard(path: str) -> Iterator[Dict[str, Any]]:
    """Yield each record from one gzip-JSONL shard, tolerating a truncated final line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A shard being written concurrently can have a torn last line; skip it.
                continue


def iter_records(root: str, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield every record under ``root`` (optionally one ``split``), shard by shard."""
    for path in shard_paths(root, split):
        for rec in iter_shard(path):
            yield rec
