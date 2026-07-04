"""Static per-card metadata catalog: a stable card-id -> dense index + tags/keywords/var-keys multi-hots.

Built once from the AgentHost ``--dump-cards`` JSON (``data/cards.json``). Two products:

* a **stable dense index** per card id (sorted order, index 0 = unknown / no card), used as the card
  embedding index — no CRC32 hash collisions, and consistent across every process that loads the same
  dump;
* a **static multi-hot row** per index (CardTags ++ canonical CardKeywords ++ declared dynamic-var
  keys). The model registers this table as a buffer and gathers the row by the card index, so the rich
  metadata lives on the GPU and never travels over the wire or into the per-step feature arrays.

Regenerate the dump with: ``Lts2.AgentHost --dump-cards > python/lts2_agent/data/cards.json``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "cards.json")


class Catalog:
    """The card index + static multi-hot table, derived deterministically from the dump."""

    def __init__(self, cards: list[dict]):
        cards = sorted(cards, key=lambda c: c["id"])
        # Index 0 is reserved for "no card" / an id absent from the catalog (combat-generated cards).
        self.card_index: dict[str, int] = {c["id"]: i + 1 for i, c in enumerate(cards)}
        self.size = len(cards) + 1

        self.tags = sorted({t for c in cards for t in c.get("tags", [])})
        self.keywords = sorted({k for c in cards for k in c.get("keywords", [])})
        self.varkeys = sorted({v for c in cards for v in c.get("varKeys", [])})
        self.static_dim = len(self.tags) + len(self.keywords) + len(self.varkeys)

        col = {}
        for i, t in enumerate(self.tags):
            col[("tag", t)] = i
        for i, k in enumerate(self.keywords):
            col[("kw", k)] = len(self.tags) + i
        for i, v in enumerate(self.varkeys):
            col[("var", v)] = len(self.tags) + len(self.keywords) + i

        table = np.zeros((self.size, self.static_dim), dtype=np.float32)
        for c in cards:
            row = self.card_index[c["id"]]
            for t in c.get("tags", []):
                table[row, col[("tag", t)]] = 1.0
            for k in c.get("keywords", []):
                table[row, col[("kw", k)]] = 1.0
            for v in c.get("varKeys", []):
                table[row, col[("var", v)]] = 1.0
        self.static_table = table

        # A compact content signature, folded into the feature-version check so a changed catalog
        # (game update -> different vocab/size) rejects stale checkpoints.
        self.signature = f"{self.size}-{len(self.tags)}-{len(self.keywords)}-{len(self.varkeys)}"

    def index_of(self, card_id: str) -> int:
        return self.card_index.get(card_id, 0)


_cache: dict[str, Catalog] = {}


def try_load(path: str = DEFAULT_PATH) -> Optional[Catalog]:
    """Load the catalog, or ``None`` if the dump is absent (fresh clone) — callers fall back to a hash."""
    if path in _cache:
        return _cache[path]
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        cat = Catalog(json.load(f))
    _cache[path] = cat
    return cat
