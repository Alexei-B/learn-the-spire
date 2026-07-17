"""Per-field reconstruction report card (roadmap 3.1 CP4 / design §8).

Given a batched decoder output and the tokenizer target arrays, compute the exact metric set the
roadmap names as the contract — card-id top-1, zone accuracy, power-id top-1, RAW-unit MAEs for power
amount / creature HP / creature block / intent damage, energy accuracy, relic/potion set-F1, hand/pile
size accuracy, pending-choice accuracy, and the aggregate exact-state rate (full decoded canonical dict
== original after detokenize-level quantization). MAEs are reported in RAW game units (symexp'd), not
symlog space.

Every metric is accumulated as a per-sample ``(numerator, denominator)`` pair so the same routine yields
both the overall number and the per-``act`` breakdown (the dashboard's group-by), by summing the pairs
within a group.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from collections import Counter

import numpy as np
import torch

from .. import tokens
from . import spec as S
from .decoder import reconstruct_arrays

# The metric names streamed to the dashboard (eval.<name>) — this list IS the CP4 contract.
METRIC_NAMES = [
    "card_id_top1", "card_zone_acc", "power_id_top1", "power_amount_mae",
    "creature_hp_mae", "creature_block_mae", "intent_damage_mae", "energy_acc",
    "relic_set_f1", "potion_set_f1", "hand_size_acc", "pile_size_acc",
    "pending_choice_acc", "exact_state_rate", "exact_mech_rate", "state_dist", "field_acc",
    "action_snr", "canon_dist",
]


# Median fraction of token-fields changed by ONE real action (state -> nextState, the same
# _state_dist metric). Re-measure with `python -m lts2_agent.wm.footprint` if the tokenizer layout or
# corpus changes materially (the field universe sets both the numerator and denominator).
#   v3 (factored population rows, zone-count vector, 2026-07-17, 3,000 val transitions): PlayCard
#      median 0.0502, EndTurn 0.2432, SelectCards 0.2707, UsePotion 0.1166, DiscardPotion 0.0043,
#      overall median 0.1224. A PlayCard now usually shifts counts between two zone columns of a single
#      shared row (draw->discard) instead of moving a whole card token, so its footprint dropped sharply.
#   v2 (count-grouped cards, zone in key, 3,000 val): PlayCard 0.1409, EndTurn 0.2684, overall 0.1704.
#   v1 (per-instance card tokens): PlayCard 0.108, EndTurn 0.213, overall 0.1303.
ACTION_FOOTPRINT = 0.1224


def _symexp_np(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.abs(y))


def _target_arrays(batch: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
    """Slice the batched tokenizer target tensors into per-sample numpy array dicts for detokenize."""
    host = {k: v.detach().cpu().numpy() for k, v in batch.items()}
    B = host["global_idx"].shape[0]
    out: List[Dict[str, np.ndarray]] = []
    for b in range(B):
        d: Dict[str, np.ndarray] = {}
        for k, v in host.items():
            d[k] = v[b]
        out.append(d)
    return out


def _state_dist(pa: Dict[str, np.ndarray], ta: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """``(mismatched, total)`` token-fields between a predicted and a target array dict.

    Every field (categorical column, integer-rounded numeric column, keyword block) of every token
    is weighted equally over the union of real and predicted slots; a slot present on only one side
    counts as fully wrong. 0/total = perfect reconstruction — the smooth companion to the
    all-or-nothing ``exact_state_rate``.
    """
    num = 0.0
    den = 0.0
    for t in S.TYPES:
        fields = len(t.cat_cols) + t.num_width + (1 if t.has_kw else 0)
        if fields == 0:
            continue
        if t.mask_key:
            tm = ta[t.mask_key].astype(bool)
            pm = pa[t.mask_key].astype(bool)
        else:
            tm = np.ones(1, dtype=bool)
            pm = np.ones(1, dtype=bool)
        both = tm & pm
        only = tm ^ pm
        num += float(only.sum()) * fields
        den += float(only.sum()) * fields + float(both.sum()) * fields
        if not both.any():
            continue
        if t.idx_key:
            pi = np.atleast_2d(pa[t.idx_key])[both]
            ti = np.atleast_2d(ta[t.idx_key])[both]
            num += float((pi != ti).sum())
        if t.num_key:
            pn = np.round(_symexp_np(np.atleast_2d(pa[t.num_key])[both]))
            tn = np.round(_symexp_np(np.atleast_2d(ta[t.num_key])[both]))
            num += float((pn != tn).sum())
        if t.has_kw:
            pk = (np.atleast_2d(pa["card_kw"])[both] >= 0.5)
            tk = (np.atleast_2d(ta["card_kw"])[both] >= 0.5)
            num += float((pk != tk).any(axis=-1).sum())   # keyword block = one field per card
    return num, max(den, 1.0)


def _canon_leaf_count(x: Any) -> int:
    """Number of leaf fields in a canonical-dict fragment (dicts recurse; lists sum; scalars = 1)."""
    if isinstance(x, dict):
        return sum(_canon_leaf_count(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_canon_leaf_count(v) for v in x) if x else 0
    return 1


def _canon_dist(tc: Any, pc: Any) -> Tuple[float, float]:
    """``(mismatched, total)`` leaf fields between two CANONICAL dicts — the tokenizer-version-
    independent reconstruction distance (both decoders emit the same canonical schema, so this is
    the cross-architecture comparator; ``state_dist`` is tokenizer-array-space and is not).

    Dicts recurse over the union of keys (a missing side counts all its leaves wrong). Lists are
    matched as content multisets: identical elements pair up, and each unpaired element counts all
    its leaves wrong (against the larger side's leaf total). Scalars compare by equality.
    """
    if isinstance(tc, dict) and isinstance(pc, dict):
        num = den = 0.0
        for k in set(tc) | set(pc):
            if k in tc and k in pc:
                n, d = _canon_dist(tc[k], pc[k])
            else:
                d = float(_canon_leaf_count(tc.get(k, pc.get(k))))
                n = d
            num += n
            den += d
        return num, den
    if isinstance(tc, (list, tuple)) and isinstance(pc, (list, tuple)):
        # Multiset match on serialized content; unpaired elements are fully wrong.
        import json as _json
        ta = Counter(_json.dumps(x, sort_keys=True, default=str) for x in tc)
        pa = Counter(_json.dumps(x, sort_keys=True, default=str) for x in pc)
        leaves_by_key: Dict[str, int] = {}
        for x in list(tc) + list(pc):
            key = _json.dumps(x, sort_keys=True, default=str)
            leaves_by_key.setdefault(key, max(1, _canon_leaf_count(x)))
        num = den = 0.0
        for key in set(ta) | set(pa):
            matched = min(ta.get(key, 0), pa.get(key, 0))
            unpaired_t = ta.get(key, 0) - matched
            unpaired_p = pa.get(key, 0) - matched
            den += matched * leaves_by_key[key]
            # Unpaired target elements are misses; unpaired predicted are spurious — both wrong.
            num += (unpaired_t + unpaired_p) * leaves_by_key[key]
            den += (unpaired_t + unpaired_p) * leaves_by_key[key]
        return num, max(den, 1.0)
    return (0.0, 1.0) if tc == pc else (1.0, 1.0)


def _set_f1(pred: List[int], tgt: List[int]) -> float:
    ps, ts = set(pred), set(tgt)
    if not ps and not ts:
        return 1.0
    inter = len(ps & ts)
    if inter == 0:
        return 0.0
    prec = inter / len(ps)
    rec = inter / len(ts)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def report_pairs(batch: Dict[str, torch.Tensor],
                 outputs: Dict[str, Dict[str, torch.Tensor]],
                 dedup: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-sample ``(numerator, denominator)`` arrays for every contract metric (length B each).

    ``dedup`` forwards to :func:`reconstruct_arrays` — a pure decode-time relic-slot dedup that lifts
    ``relic_set_f1`` (and ``state_dist``) for the slot-head model without any training change."""
    B = batch["global_idx"].shape[0]
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def slot_acc(type_name: str, col: int) -> Tuple[np.ndarray, np.ndarray]:
        t = S.TYPE_BY_NAME[type_name]
        mask = batch[t.mask_key]                                  # [B, slots]
        pred = outputs[type_name]["cat"][col].argmax(dim=-1)      # [B, slots]
        tgt = batch[t.idx_key][..., col]
        correct = ((pred == tgt) & mask).sum(dim=1).float().cpu().numpy()
        den = mask.sum(dim=1).float().cpu().numpy()
        return correct, den

    def slot_mae(type_name: str, num_col: int) -> Tuple[np.ndarray, np.ndarray]:
        t = S.TYPE_BY_NAME[type_name]
        mask = batch[t.mask_key]
        pred_raw = torch.sign(outputs[type_name]["num"][..., num_col]) * torch.expm1(
            outputs[type_name]["num"][..., num_col].abs())
        tgt_raw = torch.sign(batch[t.num_key][..., num_col]) * torch.expm1(
            batch[t.num_key][..., num_col].abs())
        err = (pred_raw - tgt_raw).abs() * mask.to(pred_raw.dtype)
        return err.sum(dim=1).cpu().numpy(), mask.sum(dim=1).float().cpu().numpy()

    def card_zone_counts_acc() -> Tuple[np.ndarray, np.ndarray]:
        """v3 card_zone_acc: fraction of present card population rows whose FULL per-zone count vector
        (the five count_<zone> numeric columns, integer-rounded) is reconstructed exactly. Replaces the
        v2 single categorical zone column — zone membership is now the count vector."""
        t = S.TYPE_BY_NAME["card"]
        mask = batch[t.mask_key]                                   # [B, slots]
        cols = S.CARD_COUNT_COLS
        pred = outputs["card"]["num"][..., cols]                  # [B, slots, 5]
        tgt = batch[t.num_key][..., cols]
        pred_i = torch.round(torch.sign(pred) * torch.expm1(pred.abs()))
        tgt_i = torch.round(torch.sign(tgt) * torch.expm1(tgt.abs()))
        match = (pred_i == tgt_i).all(dim=-1) & mask              # [B, slots]
        correct = match.sum(dim=1).float().cpu().numpy()
        den = mask.sum(dim=1).float().cpu().numpy()
        return correct, den

    pairs["card_id_top1"] = slot_acc("card", 0)
    pairs["card_zone_acc"] = card_zone_counts_acc()
    pairs["power_id_top1"] = slot_acc("power", 0)
    pairs["power_amount_mae"] = slot_mae("power", S.POWER_AMOUNT_IDX)
    pairs["creature_hp_mae"] = slot_mae("creature", S.CREATURE_HP_IDX)
    pairs["creature_block_mae"] = slot_mae("creature", S.CREATURE_BLOCK_IDX)
    pairs["intent_damage_mae"] = slot_mae("intent", S.INTENT_DAMAGE_IDX)

    # energy: exact integer match after symexp+round on the global energy field.
    pred_e = outputs["global"]["num"][:, 0, S.ENERGY_NUM_IDX]
    tgt_e = batch["global_num"][:, 0, S.ENERGY_NUM_IDX]
    pred_ei = np.round(_symexp_np(pred_e.cpu().numpy()))
    tgt_ei = np.round(_symexp_np(tgt_e.cpu().numpy()))
    pairs["energy_acc"] = ((pred_ei == tgt_ei).astype(np.float32), np.ones(B, dtype=np.float32))

    # Canonical-dict-derived metrics (relic/potion sets, sizes, pending, exact-state).
    pred_arrays = reconstruct_arrays(outputs, dedup=dedup)
    tgt_arrays = _target_arrays(batch)
    relic_f1 = np.zeros(B, np.float32); potion_f1 = np.zeros(B, np.float32)
    hand_ok = np.zeros(B, np.float32); pile_num = np.zeros(B, np.float32); pile_den = np.zeros(B, np.float32)
    pend_ok = np.zeros(B, np.float32); exact = np.zeros(B, np.float32)
    exact_mech = np.zeros(B, np.float32)
    dist_num = np.zeros(B, np.float32); dist_den = np.zeros(B, np.float32)
    cd_num = np.zeros(B, np.float32); cd_den = np.zeros(B, np.float32)
    for b in range(B):
        dist_num[b], dist_den[b] = _state_dist(pred_arrays[b], tgt_arrays[b])
        pc = tokens.detokenize(pred_arrays[b])
        tc = tokens.detokenize(tgt_arrays[b])
        relic_f1[b] = _set_f1(pc["relics"], tc["relics"])
        potion_f1[b] = _set_f1(pc["potions"], tc["potions"])
        hand_ok[b] = 1.0 if len(pc["cards"]["hand"]) == len(tc["cards"]["hand"]) else 0.0
        for z in ("draw", "discard", "exhaust"):
            pile_den[b] += 1.0
            if len(pc["cards"][z]) == len(tc["cards"][z]):
                pile_num[b] += 1.0
        pend_ok[b] = 1.0 if (pc["pending"] is None) == (tc["pending"] is None) else 0.0
        ok, _ = tokens._deep_diff("", tc, pc)
        exact[b] = 1.0 if ok else 0.0
        cd_num[b], cd_den[b] = _canon_dist(tc, pc)
        # Mechanical exactness: strict minus the run-bookkeeping integers (score/gold), which are
        # high-entropy, combat-irrelevant, and by far the hardest fields to regress to the exact
        # integer — they gate the strict metric long after the fight itself reconstructs perfectly.
        tm = dict(tc); pm = dict(pc)
        tm["global"] = {k: v for k, v in tc["global"].items() if k not in ("score", "gold")}
        pm["global"] = {k: v for k, v in pc["global"].items() if k not in ("score", "gold")}
        ok_m, _ = tokens._deep_diff("", tm, pm)
        exact_mech[b] = 1.0 if ok_m else 0.0

    ones = np.ones(B, np.float32)
    pairs["relic_set_f1"] = (relic_f1, ones)
    pairs["potion_set_f1"] = (potion_f1, ones)
    pairs["hand_size_acc"] = (hand_ok, ones)
    pairs["pile_size_acc"] = (pile_num, pile_den)
    pairs["pending_choice_acc"] = (pend_ok, ones)
    pairs["exact_state_rate"] = (exact, ones)
    pairs["exact_mech_rate"] = (exact_mech, ones)
    pairs["state_dist"] = (dist_num, dist_den)
    # The ascending complement: fraction of token-fields correctly decoded (1 - state_dist),
    # an accuracy so it pins to the 0..1 axis and suits the top-end display scales.
    pairs["field_acc"] = (dist_den - dist_num, dist_den)
    pairs["canon_dist"] = (cd_num, cd_den)
    # Signal-to-noise for the M4 gate: how many times larger is a MEDIAN action's state-change
    # footprint than the decoder's reconstruction error, in the same token-field distance.
    # SNR 1 = reconstruction noise equals a whole action; the roadmap gate is >=~4 to start the
    # predictor phase, >=~13 to trust fine-grained predictor comparisons.
    # (num, den) = (footprint * den, mismatches) so grouped sums give footprint / group-distance.
    pairs["action_snr"] = (ACTION_FOOTPRINT * dist_den, np.maximum(dist_num, 1e-9))
    return pairs


def aggregate(pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
              select: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Overall (or masked-subset) metric values from per-sample pairs. ``select`` is a bool mask."""
    out: Dict[str, float] = {}
    for name, (num, den) in pairs.items():
        n = num[select] if select is not None else num
        d = den[select] if select is not None else den
        ds = float(d.sum())
        out[name] = float(n.sum()) / ds if ds > 0 else 0.0
    return out


def merge_pairs(accum: Dict[str, Tuple[List, List]],
                pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                acts: List[Any]) -> None:
    """Append a batch's per-sample pairs (+ acts) into a running accumulator for a full-split pass."""
    for name, (num, den) in pairs.items():
        a, b = accum.setdefault(name, ([], []))
        a.append(num); b.append(den)
    accum.setdefault("_acts", ([], []))[0].append(np.asarray(acts, dtype=object))


def finalize(accum: Dict[str, Tuple[List, List]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Turn an accumulator into (overall metrics, per-act metrics)."""
    pairs = {name: (np.concatenate(a), np.concatenate(b))
             for name, (a, b) in accum.items() if name != "_acts"}
    acts = np.concatenate(accum["_acts"][0]) if "_acts" in accum else None
    overall = aggregate(pairs)
    by_act: Dict[str, Dict[str, float]] = {}
    if acts is not None:
        for act in sorted({str(x) for x in acts}):
            sel = np.asarray([str(x) == act for x in acts], dtype=bool)
            by_act[act] = aggregate(pairs, sel)
    return overall, by_act


def format_report(overall: Dict[str, float], by_act: Dict[str, Dict[str, float]],
                  n_states: int, header: str = "") -> str:
    """Render the report card as text (the CP4 artifact)."""
    lines = ["=" * 72]
    if header:
        lines.append(header)
    lines.append(f"WORLD-MODEL RECONSTRUCTION REPORT CARD   states={n_states}")
    lines.append("=" * 72)
    mae = {"power_amount_mae", "creature_hp_mae", "creature_block_mae", "intent_damage_mae"}
    for name in METRIC_NAMES:
        unit = " (raw units)" if name in mae else ""
        lines.append(f"  {name:22s} {overall[name]:8.4f}{unit}")
    if by_act:
        lines.append("-" * 72)
        lines.append("by act:")
        acts = sorted(by_act)
        head = "  " + "metric".ljust(22) + "".join(f"act{a:>6}" for a in acts)
        lines.append(head)
        for name in METRIC_NAMES:
            row = "  " + name.ljust(22) + "".join(f"{by_act[a][name]:9.3f}" for a in acts)
            lines.append(row)
    lines.append("=" * 72)
    return "\n".join(lines)
