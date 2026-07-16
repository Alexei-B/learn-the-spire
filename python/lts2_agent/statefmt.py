"""Canonical-state pretty-printer + field-level diff (roadmap 3.2, design §4.3 job 3).

The world-model decoder (M3) turns a predicted latent into a **canonical state dict** — the exact
shape :func:`lts2_agent.tokens.detokenize` produces (``{global, pending, cards{zone->[...]},
creatures[...], orbs[...], relics[...], potions[...]}``). This module is the human-readable window
onto that shape:

* :func:`format_state` — render one canonical state as compact text: creatures (player / Osty /
  enemies) with hp/block/powers/intents, energy/stars/turn, the hand with per-card cost/damage/
  block/upgrade, draw/discard/exhaust as counted multisets, relics, potions, and any pending choice.
* :func:`diff_states` — a field-level "what changed" view between two canonical states: HP/block/
  energy deltas, cards that moved zones (multiset diffs per zone), powers gained/lost/changed,
  enemies that died, and intents that changed. This is the primitive the TUI prediction inspector
  (4.4) and the predictor report card (4.3) both render.

It works on **either** ``detokenize(tokenize(raw_wire_state))`` **or** any dict already in the
canonical shape (a decoder's output). Pass a raw wire observation and it canonicalizes first
(:func:`as_canonical`).

Hashed-lossy ids
----------------
Several wire ids are hashed into fixed vocabs by the tokenizer (:data:`lts2_agent.tokens.LOSSY_FIELDS`
— monster / character / orb / enchant / affliction / keyword buckets), so a canonical dict stores a
bucket, not a name. :func:`build_hash_names` scans a corpus once and writes ``data/hash_names.json``,
a ``{bucket -> [names]}`` reverse map per vocab (collisions list every colliding name). When that map
is passed in, the printer shows names; otherwise it shows ``#<bucket>``.

CLI::

    python -m lts2_agent.statefmt build-hash-names --corpus data/corpus [--out data/hash_names.json]
    python -m lts2_agent.statefmt show  --corpus data/corpus --split val [--index N]
    python -m lts2_agent.statefmt diff  --corpus data/corpus --split val [--index N]

Stdlib + numpy only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import catalog, tokens

# ==================================================================================================
# Reverse lookups for the exactly-invertible catalog indices and the fixed enums.
# ==================================================================================================


def _card_id(idx: int) -> str:
    return tokens._CARDS.id_of(idx) or (f"#card{idx}" if idx else "-")


def _power_id(idx: int) -> str:
    return tokens._POWERS.id_of(idx) or (f"#power{idx}" if idx else "-")


def _relic_id(idx: int) -> str:
    return tokens._RELICS.id_of(idx) or (f"#relic{idx}" if idx else "-")


def _potion_id(idx: int) -> str:
    return tokens._POTIONS.id_of(idx) or (f"#potion{idx}" if idx else "-")


def _enum(idx: int, table: List[str]) -> str:
    return tokens._enum_name(idx, table)


# ==================================================================================================
# Hashed-lossy id reverse map (bucket -> name), built from the corpus. The vocabs and hashing must
# mirror tokens.py exactly so a bucket resolves to the string(s) that produced it.
# ==================================================================================================

# vocab-name -> (hashing fn, wire-string extractor description). One entry per hashed vocab.
_HASH_VOCABS = ("monster", "character", "orb", "enchant", "afflict", "keyword")


def _monster_bucket(s: str) -> int:
    return tokens._mon(s)


def _char_bucket(s: str) -> int:
    return tokens._char(s)


def _orb_bucket(s: str) -> int:
    return tokens._orb(s)


def _enchant_bucket(s: str) -> int:
    return tokens._ench(s)


def _afflict_bucket(s: str) -> int:
    return tokens._affl(s)


def _keyword_bucket(s: str) -> int:
    # Mirror tokens._kw_multi: bucket = stable_hash(k, KW_BUCKETS + 1) - 1.
    return catalog.stable_hash(s, tokens.KW_BUCKETS + 1) - 1


_BUCKET_FN = {
    "monster": _monster_bucket, "character": _char_bucket, "orb": _orb_bucket,
    "enchant": _enchant_bucket, "afflict": _afflict_bucket, "keyword": _keyword_bucket,
}


def _collect_strings(state: Dict[str, Any], acc: Dict[str, set]) -> None:
    """Harvest every raw hashed-id string from a wire state into per-vocab string sets."""
    combat = state.get("combat") or {}
    for enemy in combat.get("enemies") or []:
        if enemy.get("monsterId"):
            acc["monster"].add(enemy["monsterId"])
    for pl in state.get("players") or []:
        if pl.get("character"):
            acc["character"].add(pl["character"])
        cs = pl.get("combatState") or {}
        for orb in cs.get("orbs") or []:
            if orb.get("orbId"):
                acc["orb"].add(orb["orbId"])
        piles = ["hand", "drawPile", "discardPile", "exhaustPile"]
        for pile in piles:
            for card in cs.get(pile) or []:
                if card.get("enchantmentId"):
                    acc["enchant"].add(card["enchantmentId"])
                if card.get("afflictionId"):
                    acc["afflict"].add(card["afflictionId"])
                for kw in card.get("addedKeywords") or []:
                    acc["keyword"].add(kw)
    pc = state.get("pendingChoice") or {}
    for card in pc.get("options") or []:
        if card.get("enchantmentId"):
            acc["enchant"].add(card["enchantmentId"])
        if card.get("afflictionId"):
            acc["afflict"].add(card["afflictionId"])
        for kw in card.get("addedKeywords") or []:
            acc["keyword"].add(kw)


def build_hash_names(corpus_root: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Scan a corpus once and build the ``{vocab -> {bucket -> [names]}}`` reverse map.

    Every observed raw string is hashed with the same function the tokenizer uses; the map records,
    per bucket, the sorted set of strings that landed there. A bucket with more than one string is a
    **collision** (listed under ``"collisions"``). Includes the tokenizer signature so a stale map
    (built against a different vocab) is detectable.
    """
    from . import corpus

    acc: Dict[str, set] = {v: set() for v in _HASH_VOCABS}
    n = 0
    for rec in corpus.iter_records(corpus_root):
        for which in ("state", "nextState"):
            st = rec.get(which)
            if st:
                _collect_strings(st, acc)
        n += 1
        if limit and n >= limit:
            break

    out: Dict[str, Any] = {"signature": tokens.tokenizer_signature(), "recordsScanned": n}
    collisions: Dict[str, Dict[str, List[str]]] = {}
    for vocab in _HASH_VOCABS:
        fn = _BUCKET_FN[vocab]
        buckets: Dict[int, set] = {}
        for s in acc[vocab]:
            buckets.setdefault(fn(s), set()).add(s)
        out[vocab] = {str(b): sorted(names) for b, names in sorted(buckets.items())}
        coll = {str(b): sorted(names) for b, names in buckets.items() if len(names) > 1}
        if coll:
            collisions[vocab] = coll
    out["collisions"] = collisions
    return out


def load_hash_names(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the hash-name map from ``data/hash_names.json`` (or ``path``); ``None`` if absent."""
    if path is None:
        path = os.path.join(catalog.DATA_DIR, "hash_names.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _name(hash_names: Optional[Dict[str, Any]], vocab: str, bucket: int) -> str:
    """Resolve a hashed bucket to a name via ``hash_names``; ``#<bucket>`` when unknown, ``-`` for 0."""
    if bucket == 0:
        return "-"
    if hash_names:
        names = (hash_names.get(vocab) or {}).get(str(bucket))
        if names:
            return names[0] if len(names) == 1 else "|".join(names)
    return f"#{bucket}"


# ==================================================================================================
# Canonicalization: accept a wire observation OR an already-canonical dict.
# ==================================================================================================


def as_canonical(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``state`` unchanged if it is already canonical (has a ``global`` block), else build the
    canonical view from a raw wire observation."""
    if "global" in state and "creatures" in state:
        return state
    return tokens._canonical_from_state(state)


def from_tokens(state: Dict[str, Any]) -> Dict[str, Any]:
    """Round-trip a raw wire state through the tokenizer into canonical form (what the model sees)."""
    return tokens.detokenize(tokens.tokenize(state, strict=False))


# ==================================================================================================
# Pretty-printer.
# ==================================================================================================


def _fmt_powers(powers: List[Dict[str, Any]]) -> str:
    if not powers:
        return ""
    parts = [f"{_power_id(p['idx'])}{_signed(p['amount'])}" for p in powers]
    return "  powers: " + ", ".join(parts)


def _signed(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def _fmt_intents(intents: List[Dict[str, Any]]) -> str:
    if not intents:
        return ""
    parts = []
    for it in intents:
        t = _enum(it["type"], tokens.INTENT_TYPES)
        if it.get("hasDamage"):
            hits = it.get("hits") or 1
            parts.append(f"{t} {it['damage']}x{hits}" if hits and hits != 1 else f"{t} {it['damage']}")
        else:
            parts.append(t)
    return "  intent: " + ", ".join(parts)


def _card_line(c: Dict[str, Any], hash_names: Optional[Dict[str, Any]]) -> str:
    cid = _card_id(c["cardIndex"])
    typ = _enum(c["type"], tokens.CARD_TYPES)
    bits = [cid]
    if c.get("upgraded"):
        bits[0] = cid + "+"
    cost = "X" if c.get("costsX") else str(c.get("energyCost", 0))
    bits.append(f"c{cost}")
    if c.get("starCost", -1) not in (-1, 0):
        bits.append(f"star{c['starCost']}")
    bits.append(typ[:3])
    if c.get("hasDamage"):
        bits.append(f"dmg{c['damage']}")
    if c.get("hasBlock"):
        bits.append(f"blk{c['block']}")
    if c.get("hasSummon"):
        bits.append(f"sum{c['summon']}")
    tt = _enum(c["targetType"], tokens.TARGET_TYPES)
    if tt not in ("None", "Self"):
        bits.append(f"->{tt}")
    ench = _name(hash_names, "enchant", c.get("enchant", 0))
    if ench != "-":
        bits.append(f"ench:{ench}")
    aff = _name(hash_names, "afflict", c.get("afflict", 0))
    if aff != "-":
        bits.append(f"affl:{aff}")
    kws = [_name(hash_names, "keyword", b) for b in c.get("keywords") or []]
    if kws:
        bits.append("kw:" + ",".join(kws))
    if not c.get("canPlay", 1):
        bits.append("[X]")
    return " ".join(bits)


def _multiset_lines(cards: List[Dict[str, Any]], hash_names: Optional[Dict[str, Any]]) -> str:
    """Render a pile as a counted multiset of card signatures, e.g. ``STRIKE x4, DEFEND x5``."""
    counts: Dict[str, int] = {}
    for c in cards:
        key = _card_multiset_key(c, hash_names)
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{k} x{n}" if n > 1 else k for k, n in sorted(counts.items())]
    return "{" + ", ".join(parts) + "}"


def _card_multiset_key(c: Dict[str, Any], hash_names: Optional[Dict[str, Any]]) -> str:
    cid = _card_id(c["cardIndex"])
    if c.get("upgraded"):
        cid += "+"
    return cid


def format_state(state: Dict[str, Any], hash_names: Optional[Dict[str, Any]] = None,
                 title: str = "STATE") -> str:
    """Render a canonical (or raw wire) state as compact, readable text."""
    cv = as_canonical(state)
    g = cv["global"]
    lines: List[str] = []
    phase = _enum(g["phase"], tokens.GAME_PHASES)
    side = _enum(g["side"], tokens.SIDES)
    turn = _enum(g["turnPhase"], tokens.TURN_PHASES)
    lines.append(f"=== {title}  ({phase} / turn={turn} / side={side}) ===")
    over = "  GAME OVER" + ("(victory)" if g.get("isVictory") else "(defeat)") if g.get("isGameOver") else ""
    lines.append(
        f"act {g['act']} floor {g['floor']} | energy {g['energy']}/{g['maxEnergy']} "
        f"stars {g['stars']} | turn {g['turnNumber']} round {g['roundNumber']} | "
        f"gold {g['gold']}{over}")

    # Creatures: player, osty, then enemies.
    for cr in cv["creatures"]:
        kind = _enum(cr["kind"], tokens.CREATURE_KINDS)
        if kind == "player":
            ident = _name(hash_names, "character", cr["identity"])
            head = f"Player[{ident}]"
        elif kind == "osty":
            head = "Osty"
        else:
            ident = _name(hash_names, "monster", cr["identity"])
            alive = "" if cr["active"] else " (dead)"
            head = f"Enemy#{cr['combatId']}[{ident}]{alive}"
        block = f" blk {cr['block']}" if cr["block"] else ""
        line = f"  {head}: HP {cr['currentHp']}/{cr['maxHp']}{block}"
        line += _fmt_intents(cr["intents"]) + _fmt_powers(cr["powers"])
        lines.append(line)

    if cv["orbs"]:
        orbs = ", ".join(f"{_name(hash_names, 'orb', o['orb'])}({o['passiveValue']}/{o['evokeValue']})"
                         for o in cv["orbs"])
        lines.append(f"  Orbs: {orbs}")

    # Cards by zone.
    hand = cv["cards"]["hand"]
    lines.append(f"  Hand ({len(hand)}): " + "; ".join(_card_line(c, hash_names) for c in hand))
    for zone, label in (("draw", "Draw"), ("discard", "Discard"), ("exhaust", "Exhaust")):
        z = cv["cards"][zone]
        lines.append(f"  {label} ({len(z)}): " + _multiset_lines(z, hash_names))
    offered = cv["cards"]["offered"]
    if offered:
        lines.append(f"  Offered ({len(offered)}): " + "; ".join(_card_line(c, hash_names)
                                                                   for c in offered))

    relics = [_relic_id(r) for r in cv["relics"] if r]
    if relics:
        lines.append(f"  Relics ({len(relics)}): " + ", ".join(relics))
    potions = [_potion_id(p) for p in cv["potions"] if p]
    if potions:
        lines.append(f"  Potions ({len(potions)}): " + ", ".join(potions))

    p = cv["pending"]
    if p is not None:
        upg = " (upgrade)" if p.get("isUpgradeSelection") else ""
        lines.append(f"  Pending choice: select {p['minSelect']}..{p['maxSelect']} of "
                     f"{len(offered)} offered{upg}")
    return "\n".join(lines)


# ==================================================================================================
# Field-level diff.
# ==================================================================================================


def _creature_index(cv: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """Map each creature to a stable key: player/osty by kind, enemies by combatId."""
    out: Dict[Any, Dict[str, Any]] = {}
    for cr in cv["creatures"]:
        kind = _enum(cr["kind"], tokens.CREATURE_KINDS)
        key = kind if kind in ("player", "osty") else ("enemy", cr["combatId"])
        out[key] = cr
    return out


def _powers_by_id(cr: Dict[str, Any]) -> Dict[int, int]:
    return {p["idx"]: p["amount"] for p in cr["powers"]}


def _zone_multiset(cards: List[Dict[str, Any]], hash_names: Optional[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in cards:
        k = _card_multiset_key(c, hash_names)
        counts[k] = counts.get(k, 0) + 1
    return counts


def diff_states(a: Dict[str, Any], b: Dict[str, Any],
                hash_names: Optional[Dict[str, Any]] = None) -> str:
    """Field-level "what changed" between canonical states ``a`` -> ``b`` (each may be raw wire)."""
    ca, cb = as_canonical(a), as_canonical(b)
    lines: List[str] = ["=== DIFF (a -> b) ==="]

    # Globals: energy, stars, turn.
    ga, gb = ca["global"], cb["global"]
    gparts = []
    for k, label in (("energy", "energy"), ("stars", "stars"), ("turnNumber", "turn"),
                     ("gold", "gold")):
        if ga.get(k) != gb.get(k):
            gparts.append(f"{label} {ga.get(k)}->{gb.get(k)}")
    if ga.get("phase") != gb.get("phase"):
        gparts.append(f"phase {_enum(ga['phase'], tokens.GAME_PHASES)}->"
                      f"{_enum(gb['phase'], tokens.GAME_PHASES)}")
    if gparts:
        lines.append("  Global: " + ", ".join(gparts))

    # Creatures matched by key.
    ia, ib = _creature_index(ca), _creature_index(cb)
    for key in list(ia) + [k for k in ib if k not in ia]:
        ca_cr, cb_cr = ia.get(key), ib.get(key)
        label = _creature_label(key, ca_cr or cb_cr, hash_names)
        if ca_cr and not cb_cr:
            lines.append(f"  {label}: REMOVED")
            continue
        if cb_cr and not ca_cr:
            lines.append(f"  {label}: ADDED (HP {cb_cr['currentHp']}/{cb_cr['maxHp']})")
            continue
        cparts = []
        if ca_cr["currentHp"] != cb_cr["currentHp"]:
            d = cb_cr["currentHp"] - ca_cr["currentHp"]
            cparts.append(f"HP {ca_cr['currentHp']}->{cb_cr['currentHp']} ({_signed(d)})")
        if ca_cr["block"] != cb_cr["block"]:
            d = cb_cr["block"] - ca_cr["block"]
            cparts.append(f"block {ca_cr['block']}->{cb_cr['block']} ({_signed(d)})")
        if ca_cr["active"] and not cb_cr["active"]:
            cparts.append("DIED")
        elif not ca_cr["active"] and cb_cr["active"]:
            cparts.append("REVIVED")
        # Powers.
        pa, pb = _powers_by_id(ca_cr), _powers_by_id(cb_cr)
        for pid in sorted(set(pa) | set(pb)):
            if pid not in pa:
                cparts.append(f"+{_power_id(pid)}({pb[pid]})")
            elif pid not in pb:
                cparts.append(f"-{_power_id(pid)}")
            elif pa[pid] != pb[pid]:
                cparts.append(f"{_power_id(pid)} {pa[pid]}->{pb[pid]}")
        # Intents.
        ta, tb = _intent_sig(ca_cr), _intent_sig(cb_cr)
        if ta != tb:
            cparts.append(f"intent {ta or '-'} -> {tb or '-'}")
        if cparts:
            lines.append(f"  {label}: " + ", ".join(cparts))

    # Cards moved between zones (per-zone multiset diff).
    for zone, zlabel in (("hand", "hand"), ("draw", "draw"), ("discard", "discard"),
                         ("exhaust", "exhaust"), ("offered", "offered")):
        ma = _zone_multiset(ca["cards"][zone], hash_names)
        mb = _zone_multiset(cb["cards"][zone], hash_names)
        added = _multiset_delta(mb, ma)
        removed = _multiset_delta(ma, mb)
        if added or removed:
            parts = []
            if removed:
                parts.append("-" + ", -".join(f"{k}x{v}" if v > 1 else k for k, v in removed))
            if added:
                parts.append("+" + ", +".join(f"{k}x{v}" if v > 1 else k for k, v in added))
            lines.append(f"  {zlabel}: " + " ".join(parts))

    # Relics / potions.
    ra = _count([_relic_id(r) for r in ca["relics"] if r])
    rb = _count([_relic_id(r) for r in cb["relics"] if r])
    if ra != rb:
        added = _multiset_delta(rb, ra)
        removed = _multiset_delta(ra, rb)
        parts = []
        if removed:
            parts.append("-" + ", -".join(k for k, _ in removed))
        if added:
            parts.append("+" + ", +".join(k for k, _ in added))
        lines.append("  relics: " + " ".join(parts))
    pa = _count([_potion_id(p) for p in ca["potions"] if p])
    pb = _count([_potion_id(p) for p in cb["potions"] if p])
    if pa != pb:
        added = _multiset_delta(pb, pa)
        removed = _multiset_delta(pa, pb)
        parts = []
        if removed:
            parts.append("-" + ", -".join(k for k, _ in removed))
        if added:
            parts.append("+" + ", +".join(k for k, _ in added))
        lines.append("  potions: " + " ".join(parts))

    if len(lines) == 1:
        lines.append("  (no changes)")
    return "\n".join(lines)


def _intent_sig(cr: Dict[str, Any]) -> str:
    parts = []
    for it in cr["intents"]:
        t = _enum(it["type"], tokens.INTENT_TYPES)
        if it.get("hasDamage"):
            hits = it.get("hits") or 1
            parts.append(f"{t}{it['damage']}x{hits}")
        else:
            parts.append(t)
    return ",".join(parts)


def _creature_label(key: Any, cr: Dict[str, Any], hash_names: Optional[Dict[str, Any]]) -> str:
    if key == "player":
        return f"Player[{_name(hash_names, 'character', cr['identity'])}]"
    if key == "osty":
        return "Osty"
    return f"Enemy#{key[1]}[{_name(hash_names, 'monster', cr['identity'])}]"


def _count(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def _multiset_delta(a: Dict[str, int], b: Dict[str, int]) -> List[Tuple[str, int]]:
    """Items in ``a`` beyond ``b`` (positive multiset difference), sorted by key."""
    out = []
    for k in sorted(a):
        d = a[k] - b.get(k, 0)
        if d > 0:
            out.append((k, d))
    return out


# ==================================================================================================
# CLI.
# ==================================================================================================


def _nth_record(root: str, split: Optional[str], index: int) -> Optional[Dict[str, Any]]:
    from . import corpus
    n = 0
    for rec in corpus.iter_records(root, split):
        if n == index:
            return rec
        n += 1
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Canonical-state pretty-printer / diff / hash-name map.")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build-hash-names", help="scan a corpus and write the bucket->name map")
    b.add_argument("--corpus", required=True)
    b.add_argument("--out", default=None, help="output path (default data/hash_names.json)")
    b.add_argument("--limit", type=int, default=None)

    s = sub.add_parser("show", help="pretty-print one corpus record's state")
    s.add_argument("--corpus", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--index", type=int, default=0)

    d = sub.add_parser("diff", help="diff one corpus record's state -> nextState")
    d.add_argument("--corpus", required=True)
    d.add_argument("--split", default="val")
    d.add_argument("--index", type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == "build-hash-names":
        out = args.out or os.path.join(catalog.DATA_DIR, "hash_names.json")
        m = build_hash_names(args.corpus, args.limit)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=0, sort_keys=True)
        total = sum(len(m[v]) for v in _HASH_VOCABS)
        ncoll = sum(len(v) for v in m["collisions"].values())
        print(f"wrote {out}: {m['recordsScanned']} records scanned, {total} buckets across "
              f"{len(_HASH_VOCABS)} vocabs, {ncoll} colliding buckets")
        for v in _HASH_VOCABS:
            print(f"  {v:10s} {len(m[v]):4d} buckets"
                  + (f"  ({len(m['collisions'].get(v, {}))} collisions)"
                     if m["collisions"].get(v) else ""))
        return 0

    if args.cmd in ("show", "diff"):
        hn = load_hash_names()
        rec = _nth_record(args.corpus, args.split, args.index)
        if rec is None:
            print("no such record")
            return 1
        if args.cmd == "show":
            print(format_state(rec["state"], hn))
        else:
            print(format_state(rec["state"], hn, title="STATE"))
            print()
            if rec.get("nextState"):
                print(diff_states(rec["state"], rec["nextState"], hn))
            else:
                print("(no nextState)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
