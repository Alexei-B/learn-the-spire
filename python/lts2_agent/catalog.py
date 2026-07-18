"""Shared static-metadata catalogs for the world-model tokenizer (cards / powers / relics / potions).

Generalizes the :mod:`lts2_agent.card_catalog` pattern to every dumped entity type. Each catalog is
built once from an AgentHost dump (``data/<kind>.json`` produced by ``--dump-cards`` / ``--dump-powers``
/ ``--dump-relics`` / ``--dump-potions``) and gives the tokenizer three things:

* a **stable dense index** per entity id (sorted id order, index ``0`` reserved for "none / unknown"),
  used as the entity's embedding index and — crucially for the round-trip validator — an exact inverse
  :meth:`EntityCatalog.id_of`;
* a **static feature table** ``[size, static_dim]`` (categorical one-hots ++ boolean flags ++ multi-hot
  list fields such as tags/keywords/var-keys), which the model gathers by index so the rich static
  metadata never travels over the wire or into the per-token dynamic arrays;
* a compact **content signature**, folded into ``TOKENIZER_VERSION`` stamping so a changed catalog
  (a game update growing the vocab) rejects stale checkpoints/corpora loudly.

Null-tolerant: on a fresh clone with no dump present, :func:`load` returns a :class:`HashFallback`
that maps ids into a fixed vocab by stable CRC32 (like ``card_catalog`` does) — lossy, but keeps the
tokenizer importable and runnable everywhere.
"""

from __future__ import annotations

import json
import os
import zlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Fixed vocab size used by the hashing fallback (index 0 = none/unknown; 1..N-1 = crc32 buckets).
FALLBACK_VOCAB = 4096


def stable_hash(text: str, n: int) -> int:
    """Stable (cross-process) CRC32 hash of ``text`` into ``[1, n)`` — index 0 stays reserved for none."""
    if not text:
        return 0
    return 1 + (zlib.crc32(text.encode("utf-8")) % (n - 1))


# --------------------------------------------------------------------------------------------------
# Per-kind dump schema: which fields are categorical one-hots, boolean flags, and multi-hot lists.
# --------------------------------------------------------------------------------------------------

class CatalogSpec:
    """Declares how one entity kind's dump rows become a static feature table."""

    def __init__(self, kind: str, categorical: Sequence[str], boolean: Sequence[str],
                 multihot: Sequence[str]):
        self.kind = kind
        self.categorical = list(categorical)
        self.boolean = list(boolean)
        self.multihot = list(multihot)


SPECS: Dict[str, CatalogSpec] = {
    "cards": CatalogSpec("cards", categorical=("type", "rarity", "category"),
                         boolean=("colorless", "curse", "status"),
                         multihot=("tags", "keywords", "varKeys")),
    "powers": CatalogSpec("powers", categorical=("type", "stackType", "instanceType"),
                          boolean=("allowNegative",), multihot=("varKeys",)),
    "relics": CatalogSpec("relics", categorical=("rarity", "pool"),
                          boolean=("stackable", "spawnsPets", "addsPet", "uponPickup",
                                   "showCounter", "allowedInShops"),
                          multihot=("varKeys",)),
    "potions": CatalogSpec("potions", categorical=("rarity", "usage", "targetType", "pool"),
                           boolean=("canBeGeneratedInCombat",), multihot=("varKeys",)),
}


class EntityCatalog:
    """A stable id<->index map + static multi-hot table for one entity kind, from its dump rows."""

    def __init__(self, kind: str, rows: List[dict]):
        spec = SPECS[kind]
        self.kind = kind
        rows = sorted(rows, key=lambda r: r["id"])
        self._ids: List[str] = [r["id"] for r in rows]
        # Index 0 reserved for "none / id absent from the catalog" (e.g. combat-generated content).
        self.index: Dict[str, int] = {rid: i + 1 for i, rid in enumerate(self._ids)}
        self.size = len(self._ids) + 1
        # Per-id PRINTED (static, catalog-authored) keyword name lists — the tokenizer v7 keyword channel
        # unions these with the wire's runtime addedKeywords to form each card's ABSOLUTE keyword state.
        # (Only cards carry a ``keywords`` field; other kinds keep an empty map, harmlessly.)
        self._keywords_by_id: Dict[str, List[str]] = {
            r["id"]: [str(v) for v in (r.get("keywords") or [])] for r in rows}

        # Column layout: categorical one-hots, then booleans, then multi-hot list values.
        col: Dict[Tuple[str, str], int] = {}
        self._cat_values: Dict[str, List[str]] = {}
        self._multi_values: Dict[str, List[str]] = {}
        c = 0
        for field in spec.categorical:
            vals = sorted({str(r.get(field, "")) for r in rows})
            self._cat_values[field] = vals
            for v in vals:
                col[("cat", field + "=" + v)] = c
                c += 1
        for field in spec.boolean:
            col[("bool", field)] = c
            c += 1
        for field in spec.multihot:
            vals = sorted({v for r in rows for v in (r.get(field) or [])})
            self._multi_values[field] = vals
            for v in vals:
                col[("multi", field + "=" + v)] = c
                c += 1
        self.static_dim = c

        table = np.zeros((self.size, self.static_dim), dtype=np.float32)
        for r in rows:
            row = self.index[r["id"]]
            for field in spec.categorical:
                table[row, col[("cat", field + "=" + str(r.get(field, "")))]] = 1.0
            for field in spec.boolean:
                if r.get(field):
                    table[row, col[("bool", field)]] = 1.0
            for field in spec.multihot:
                for v in (r.get(field) or []):
                    table[row, col[("multi", field + "=" + v)]] = 1.0
        self.static_table = table

        # Content signature: size + static width + per-field vocab counts, so any vocab drift is caught.
        parts = [str(self.size), str(self.static_dim)]
        for field in spec.categorical:
            parts.append(f"{field}:{len(self._cat_values[field])}")
        for field in spec.multihot:
            parts.append(f"{field}:{len(self._multi_values[field])}")
        self.signature = kind + "-" + "-".join(parts)

    def index_of(self, entity_id: Optional[str]) -> int:
        if not entity_id:
            return 0
        return self.index.get(entity_id, 0)

    def id_of(self, index: int) -> str:
        """Inverse of :meth:`index_of` for round-trip; ``""`` for 0 or an id absent from the catalog."""
        if index <= 0 or index > len(self._ids):
            return ""
        return self._ids[index - 1]

    @property
    def ids(self) -> List[str]:
        """The catalog's entity ids in stable (sorted) order (index i+1 == ids[i])."""
        return list(self._ids)

    def printed_keywords(self, entity_id: Optional[str]) -> List[str]:
        """The PRINTED (static) keyword names of a card id (empty for none/unknown or a kind with no
        ``keywords`` field). Source of the printed half of the tokenizer's absolute keyword state."""
        if not entity_id:
            return []
        return self._keywords_by_id.get(entity_id, [])


class HashFallback:
    """Drop-in stand-in for :class:`EntityCatalog` when a dump is absent: CRC32 hashing, no static table.

    Lossy (no exact inverse), so the tokenizer treats a hash-fallback catalog's ids as covered-lossy.
    """

    def __init__(self, kind: str, vocab: int = FALLBACK_VOCAB):
        self.kind = kind
        self.size = vocab
        self.static_dim = 0
        self.static_table = np.zeros((vocab, 0), dtype=np.float32)
        self.signature = kind + "-hash"

    def index_of(self, entity_id: Optional[str]) -> int:
        return stable_hash(entity_id or "", self.size)

    def id_of(self, index: int) -> str:
        return ""  # not invertible under hashing

    @property
    def ids(self) -> List[str]:
        return []  # no dump: no enumerable id list

    def printed_keywords(self, entity_id: Optional[str]) -> List[str]:
        return []  # no dump: printed keywords unknown (absolute state degrades to addedKeywords only)


Catalog = Union[EntityCatalog, HashFallback]

_cache: Dict[str, Catalog] = {}


def load(kind: str, path: Optional[str] = None) -> Catalog:
    """Load the ``kind`` catalog from its dump, or a :class:`HashFallback` if the dump is absent."""
    if kind not in SPECS:
        raise KeyError(f"unknown catalog kind {kind!r}; known: {sorted(SPECS)}")
    if path is None:
        path = os.path.join(DATA_DIR, f"{kind}.json")
    cache_key = kind + "|" + path
    if cache_key in _cache:
        return _cache[cache_key]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cat: Catalog = EntityCatalog(kind, json.load(f))
    else:
        cat = HashFallback(kind)
    _cache[cache_key] = cat
    return cat
