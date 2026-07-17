"""Set-transformer actor-critic over the entity tokenizer (roadmap 2.2, design §4.1-4.2, §6.A).

The model-free "representation upgrade" the design predicts: swap the hand-crafted scalar features
(:mod:`lts2_agent.features`) for the entity tokenizer (:mod:`lts2_agent.tokens`) under the *existing*
PPO head. Where :mod:`lts2_agent.model_torch` mean-pools a fixed scalar vector, this encodes the state
as a set of typed tokens and runs attention over them, so structural facts ("Unleash scales with
Osty's HP") become attention patterns rather than reward-correlation ghosts.

Pipeline (all shapes carry a leading batch dim ``B``):

* **Per-type embedders** turn each token kind's ``*_idx`` (catalog/enum indices) + ``*_num`` (symlog
  numerics) + gathered static catalog rows into a shared ``d_model`` vector. The *card* embedder is
  shared between state card tokens and legal-option cards, so action understanding reuses state
  understanding (design §4.4).
* **Creatures fold in their powers/intents** by scatter-add (the "powers as child tokens" of §4.1),
  then a self-attention layer contextualizes the (few) creatures against each other. These per-creature
  embeddings are what targeted options attend into.
* **Attention pooling** (Perceiver/Set-Transformer inducing points): a small set of learned latent
  queries cross-attends over the whole token set to produce a pooled state context ``z``.
* **Action scoring:** each legal option is ``(kind, entity, target)`` — a kind embedding, the option's
  card/potion embedding (card path shared with the tokenizer), and the *target creature's* contextual
  embedding gathered by ``targetCombatId -> creature slot`` — scored against ``z`` with a masked softmax,
  exactly like :mod:`lts2_agent.model_torch`.
* **Value head:** a tanh-bounded ``±value_scale`` head on ``z`` (the ±20 clamp exists so the critic
  can't run away and drag the shared trunk into divergence — see model_torch's comment).

Checkpoints are stamped with :func:`lts2_agent.tokens.tokenizer_signature` (TOKENIZER_VERSION + the four
catalog signatures); a mismatch rejects loudly, exactly like ``model_torch`` does with ``FEATURE_VERSION``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import catalog, tokens

NEG_INF = -1e9

# Padded option-set width (measured combat max ~21; matches features.MAX_OPTIONS slack).
MAX_OPTIONS = 64

# Option kinds the scorer distinguishes (combat kinds; anything else -> the trailing "Other" slot).
OPTION_KINDS = ["PlayCard", "EndTurn", "UsePotion", "DiscardPotion", "SelectCards", "Other"]
OPTION_KIND_ID = {k: i for i, k in enumerate(OPTION_KINDS)}
# The kinds the policy is actually allowed to pick during a fight (the same combat scope as model_torch).
COMBAT_OPTION_KINDS = {"PlayCard", "EndTurn", "UsePotion", "DiscardPotion"}

# State token arrays consumed by the model (TOKEN_KEYS minus the scalar token_type_* constants, which
# the model supplies itself from the fixed type embedding).
STATE_KEYS = tuple(k for k in tokens.TOKEN_KEYS if not k.startswith("token_type_"))
OPTION_KEYS = ("opt_kind", "opt_card_idx", "opt_card_num", "opt_card_kw", "opt_card_present",
               "opt_potion_idx", "opt_potion_present", "opt_target_slot", "opt_mask")
MODEL_KEYS = STATE_KEYS + OPTION_KEYS

_INT_KEYS = {"global_idx", "card_idx", "creature_idx", "power_idx", "intent_idx", "orb_idx",
             "relic_idx", "potion_idx", "opt_kind", "opt_card_idx", "opt_potion_idx",
             "opt_target_slot"}
_BOOL_KEYS = {"card_mask", "creature_mask", "power_mask", "intent_mask", "orb_mask", "relic_mask",
              "potion_mask", "opt_mask"}


# ==================================================================================================
# Featurize: (state, options) -> the model-input array dict (single obs; batch with :func:`stack`).
# ==================================================================================================

_POTIONS_CAT = catalog.load("potions")


def _creature_slot_map(state: Dict[str, Any]) -> Dict[int, int]:
    """Map an enemy ``combatId`` -> its creature slot index, in the exact order the tokenizer lays
    creatures out (players, then each player's osty, then enemies). Player/osty use combatId 0 and are
    never option targets, so only enemy ids need mapping."""
    slot = 0
    out: Dict[int, int] = {}
    players = state.get("players") or []
    for pl in players:
        cs = pl.get("combatState") or {}
        slot += 1                       # player creature
        if cs.get("osty") is not None:
            slot += 1                   # osty creature
    combat = state.get("combat") or {}
    for enemy in (combat.get("enemies") or []):
        cid = enemy.get("combatId")
        if cid is not None and slot < tokens.MAX_CREATURES:
            out[int(cid)] = slot
        slot += 1
    return out


def _encode_options(state: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Encode the legal options into fixed-width padded arrays. Card fields reuse the tokenizer's card
    featurization (identical layout to state card tokens) for consistency; targets resolve to a creature
    slot so a targeted option can attend into that creature's embedding."""
    slot_of = _creature_slot_map(state)
    n_card_idx, n_card_num = len(tokens.CARD_IDX), len(tokens.CARD_NUM)

    opt_kind = np.zeros(MAX_OPTIONS, dtype=np.int64)
    opt_card_idx = np.zeros((MAX_OPTIONS, n_card_idx), dtype=np.int64)
    opt_card_num = np.zeros((MAX_OPTIONS, n_card_num), dtype=np.float32)
    opt_card_kw = np.zeros((MAX_OPTIONS, tokens.KW_BUCKETS), dtype=np.float32)
    opt_card_present = np.zeros(MAX_OPTIONS, dtype=np.float32)
    opt_potion_idx = np.zeros(MAX_OPTIONS, dtype=np.int64)
    opt_potion_present = np.zeros(MAX_OPTIONS, dtype=np.float32)
    opt_target_slot = np.full(MAX_OPTIONS, -1, dtype=np.int64)
    opt_mask = np.zeros(MAX_OPTIONS, dtype=bool)

    for i, opt in enumerate(options[:MAX_OPTIONS]):
        kind = opt.get("kind", "")
        opt_kind[i] = OPTION_KIND_ID.get(kind, OPTION_KIND_ID["Other"])
        opt_mask[i] = True

        card = opt.get("card")
        if card:
            cv = tokens._card_canonical(card, "hand")
            opt_card_idx[i] = [cv[k] for k in tokens.CARD_IDX]
            opt_card_num[i] = [tokens._num_field(cv, k) for k in tokens.CARD_NUM]
            for b in cv["keywords"]:
                opt_card_kw[i, b] = 1.0
            opt_card_present[i] = 1.0

        pid = opt.get("potionId")
        if pid:
            opt_potion_idx[i] = _POTIONS_CAT.index_of(pid)
            opt_potion_present[i] = 1.0

        tgt = opt.get("targetCombatId")
        if tgt is not None and int(tgt) in slot_of:
            opt_target_slot[i] = slot_of[int(tgt)]

    return {"opt_kind": opt_kind, "opt_card_idx": opt_card_idx, "opt_card_num": opt_card_num,
            "opt_card_kw": opt_card_kw, "opt_card_present": opt_card_present,
            "opt_potion_idx": opt_potion_idx, "opt_potion_present": opt_potion_present,
            "opt_target_slot": opt_target_slot, "opt_mask": opt_mask}


def featurize(state: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """All model inputs for one observation (unbatched), keyed by :data:`MODEL_KEYS`.

    State tokens come from :func:`lts2_agent.tokens.tokenize` (non-strict: an overflowing state truncates
    rather than raising, so serving/rollout never dies on a pathological fight); options are encoded
    here. Illegal-for-combat option kinds are masked out (same policy scope as model_torch)."""
    tok = tokens.tokenize(state, strict=False)
    out: Dict[str, np.ndarray] = {k: tok[k] for k in STATE_KEYS}
    opt = _encode_options(state, options)
    # Mask non-combat option kinds so the exploring/served policy never picks a post-combat option.
    for j, o in enumerate(options[:MAX_OPTIONS]):
        if o.get("kind") not in COMBAT_OPTION_KINDS:
            opt["opt_mask"][j] = False
    out.update(opt)
    return out


def stack(feats_list: List[dict]) -> Dict[str, np.ndarray]:
    """Stack a list of single-obs feature dicts into ``[B, ...]`` arrays."""
    return {k: np.stack([f[k] for f in feats_list]) for k in MODEL_KEYS}


def to_tensors(feats_stacked: dict, device) -> Dict[str, torch.Tensor]:
    """Move a dict of stacked numpy arrays to model tensors on ``device`` (kept as a dict; the token
    model takes keyword args, unlike model_torch's positional tuple)."""
    out: Dict[str, torch.Tensor] = {}
    for k in MODEL_KEYS:
        v = np.asarray(feats_stacked[k])
        if k in _INT_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        elif k in _BOOL_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


# ==================================================================================================
# Embedding helpers.
# ==================================================================================================

def _enum_size(table: List[str]) -> int:
    """Embedding rows for a fixed enum: one per value + one for the reserved trailing UNKNOWN slot."""
    return len(table) + 1


class _MultiEmbed(nn.Module):
    """Sum of per-column embeddings for an ``[..., C]`` integer index tensor."""

    def __init__(self, sizes: List[int], dim: int):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(n, dim) for n in sizes)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # idx [..., C] long
        out = self.embs[0](idx[..., 0])
        for c in range(1, len(self.embs)):
            out = out + self.embs[c](idx[..., c])
        return out


# ==================================================================================================
# The model.
# ==================================================================================================

class TokenActorCritic(nn.Module):
    def __init__(self, d_model: int = 160, n_heads: int = 4, n_pool_layers: int = 2,
                 n_latents: int = 8, cat_dim: int = 16, ff_mult: int = 2,
                 static_tables: Optional[Dict[str, np.ndarray]] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_pool_layers = n_pool_layers
        self.n_latents = n_latents
        self.cat_dim = cat_dim

        if static_tables is None:
            static_tables = {k: catalog.load(k).static_table for k in ("cards", "powers", "relics",
                                                                        "potions")}
        for k, tbl in static_tables.items():
            self.register_buffer(f"static_{k}", torch.from_numpy(np.asarray(tbl, dtype=np.float32)))
        self._static_dim = {k: int(np.asarray(static_tables[k]).shape[1]) for k in static_tables}

        cd = cat_dim
        # --- Card embedder (shared: state card tokens AND option cards) -------------------------------
        # v3: `zone` is no longer a card categorical column (it moved into the per-zone count vector in
        # CARD_NUM), so card_sizes tracks CARD_IDX = [cardIndex, type, rarity, targetType, enchant,
        # afflict].
        card_sizes = [catalog.load("cards").size,
                      _enum_size(tokens.CARD_TYPES), _enum_size(tokens.CARD_RARITIES),
                      _enum_size(tokens.TARGET_TYPES), tokens.ENCHANT_VOCAB, tokens.AFFLICT_VOCAB]
        self.card_cat = _MultiEmbed(card_sizes, cd)
        card_in = cd + self._static_dim["cards"] + len(tokens.CARD_NUM) + tokens.KW_BUCKETS
        self.card_proj = nn.Linear(card_in, d_model)

        # --- Creature embedder -----------------------------------------------------------------------
        self.creature_cat = _MultiEmbed([_enum_size(tokens.CREATURE_KINDS), tokens.MONSTER_VOCAB], cd)
        self.creature_proj = nn.Linear(cd + len(tokens.CREATURE_NUM), d_model)

        # --- Power / intent embedders (folded into their parent creature) ----------------------------
        self.power_emb = nn.Embedding(catalog.load("powers").size, cd)
        self.power_proj = nn.Linear(cd + self._static_dim["powers"] + len(tokens.POWER_NUM), d_model)
        self.intent_emb = nn.Embedding(_enum_size(tokens.INTENT_TYPES), cd)
        self.intent_proj = nn.Linear(cd + len(tokens.INTENT_NUM), d_model)

        # --- Orb / relic / potion / global embedders -------------------------------------------------
        self.orb_emb = nn.Embedding(tokens.ORB_VOCAB, cd)
        self.orb_proj = nn.Linear(cd + len(tokens.ORB_NUM), d_model)
        self.relic_emb = nn.Embedding(catalog.load("relics").size, cd)
        self.relic_proj = nn.Linear(cd + self._static_dim["relics"], d_model)
        self.potion_emb = nn.Embedding(catalog.load("potions").size, cd)
        self.potion_proj = nn.Linear(cd + self._static_dim["potions"], d_model)
        self.global_cat = _MultiEmbed([_enum_size(tokens.GAME_PHASES), _enum_size(tokens.SIDES),
                                       _enum_size(tokens.TURN_PHASES)], cd)
        self.global_proj = nn.Linear(cd + len(tokens.GLOBAL_NUM) + 4, d_model)  # +4 pending fields

        # Token-type embedding (added to every token so attention can tell types apart).
        self.type_emb = nn.Embedding(len(tokens.TOKEN_TYPES), d_model)

        # --- Creature self-attention (contextualize the few creatures) -------------------------------
        self.creature_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.creature_norm1 = nn.LayerNorm(d_model)
        self.creature_ff = _ff(d_model, ff_mult)
        self.creature_norm2 = nn.LayerNorm(d_model)

        # --- Attention-pool: learned latent queries cross-attend over the token set ------------------
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.pool_layers = nn.ModuleList(_PoolLayer(d_model, n_heads, ff_mult) for _ in range(n_pool_layers))
        self.z_norm = nn.LayerNorm(d_model)

        # --- Option scorer: [kind ⊕ entity ⊕ target ⊕ z] -> logit ------------------------------------
        self.opt_kind_emb = nn.Embedding(len(OPTION_KINDS), d_model)
        self.o1 = nn.Linear(4 * d_model, d_model)
        self.o2 = nn.Linear(d_model, d_model)
        self.o_logit = nn.Linear(d_model, 1)

        # --- Value head (tanh-bounded; see model_torch's ±20 comment) --------------------------------
        self.value_scale = 20.0
        self.v1 = nn.Linear(d_model, d_model)
        self.v_out = nn.Linear(d_model, 1)

    # --- token embedders -------------------------------------------------------------------------

    def _embed_cards(self, card_idx, card_num, card_kw):
        cat = self.card_cat(card_idx)                                   # [.., cd]
        static = self.static_cards[card_idx[..., 0]]                    # gather by cardIndex
        x = torch.cat([cat, static, card_num, card_kw], dim=-1)
        return self.card_proj(x)

    def _embed_creatures(self, creature_idx, creature_num):
        cat = self.creature_cat(creature_idx)
        return self.creature_proj(torch.cat([cat, creature_num], dim=-1))

    def forward(self, *, global_idx, global_num, pending, card_idx, card_num, card_kw, card_mask,
                creature_idx, creature_num, creature_mask, power_idx, power_num, power_mask,
                intent_idx, intent_num, intent_mask, orb_idx, orb_num, orb_mask, relic_idx, relic_mask,
                potion_idx, potion_mask, opt_kind, opt_card_idx, opt_card_num, opt_card_kw,
                opt_card_present, opt_potion_idx, opt_potion_present, opt_target_slot, opt_mask):
        B = global_idx.shape[0]
        d = self.d_model
        te = self.type_emb.weight  # [n_types, d]

        # Global token [B,1,d].
        g = self.global_cat(global_idx)                                 # [B,1,cd]
        g = self.global_proj(torch.cat([g, global_num, pending], dim=-1))
        g = g + te[tokens.TOKEN_TYPE_ID["global"]]

        # Card tokens [B,C,d].
        cards = self._embed_cards(card_idx, card_num, card_kw) + te[tokens.TOKEN_TYPE_ID["card"]]

        # Creature tokens, then fold in powers/intents by parent slot (scatter-add) [B,K,d].
        creatures = self._embed_creatures(creature_idx, creature_num) + te[tokens.TOKEN_TYPE_ID["creature"]]

        pw = self.power_emb(power_idx[..., 0])
        pw_static = self.static_powers[power_idx[..., 0]]
        pw = self.power_proj(torch.cat([pw, pw_static, power_num], dim=-1))
        pw = pw * power_mask.unsqueeze(-1).to(pw.dtype)
        creatures = _scatter_add_by_parent(creatures, pw, power_idx[..., 1], tokens.MAX_CREATURES)

        it = self.intent_emb(intent_idx[..., 0])
        it = self.intent_proj(torch.cat([it, intent_num], dim=-1))
        it = it * intent_mask.unsqueeze(-1).to(it.dtype)
        creatures = _scatter_add_by_parent(creatures, it, intent_idx[..., 1], tokens.MAX_CREATURES)

        # Contextualize creatures against each other (self-attention over the few creature slots).
        cpad = ~creature_mask                                            # True = ignore
        ca, _ = self.creature_attn(creatures, creatures, creatures, key_padding_mask=cpad)
        creatures = self.creature_norm1(creatures + ca)
        creatures = self.creature_norm2(creatures + self.creature_ff(creatures))
        creatures = creatures * creature_mask.unsqueeze(-1).to(creatures.dtype)

        # Orb / relic / potion tokens.
        orbs = self.orb_proj(torch.cat([self.orb_emb(orb_idx[..., 0]), orb_num], dim=-1))
        orbs = orbs + te[tokens.TOKEN_TYPE_ID["orb"]]
        relics = self.relic_proj(torch.cat([self.relic_emb(relic_idx[..., 0]),
                                            self.static_relics[relic_idx[..., 0]]], dim=-1))
        relics = relics + te[tokens.TOKEN_TYPE_ID["relic"]]
        potions = self.potion_proj(torch.cat([self.potion_emb(potion_idx[..., 0]),
                                             self.static_potions[potion_idx[..., 0]]], dim=-1))
        potions = potions + te[tokens.TOKEN_TYPE_ID["potion"]]

        # Assemble the full token set + padding mask (global always valid so no all-masked row).
        parts = [g, cards, creatures, orbs, relics, potions]
        gmask = torch.ones(B, 1, dtype=torch.bool, device=g.device)
        masks = [gmask, card_mask, creature_mask, orb_mask, relic_mask, potion_mask]
        toks = torch.cat(parts, dim=1)                                  # [B,T,d]
        key_pad = ~torch.cat(masks, dim=1)                              # [B,T] True=ignore

        # Attention pooling: latents cross-attend over the token set, then self-attend.
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)               # [B,L,d]
        for layer in self.pool_layers:
            lat = layer(lat, toks, key_pad)
        z = self.z_norm(lat.mean(dim=1))                                # [B,d]

        # --- Option scoring --------------------------------------------------------------------------
        M = opt_kind.shape[1]
        kind_e = self.opt_kind_emb(opt_kind)                            # [B,M,d]
        card_e = self._embed_cards(opt_card_idx, opt_card_num, opt_card_kw)
        pot_e = self.potion_proj(torch.cat([self.potion_emb(opt_potion_idx),
                                           self.static_potions[opt_potion_idx]], dim=-1))
        entity = card_e * opt_card_present.unsqueeze(-1) + pot_e * opt_potion_present.unsqueeze(-1)

        # Gather the target creature's contextual embedding by slot (-1 -> zeros).
        has_tgt = (opt_target_slot >= 0)
        safe_slot = opt_target_slot.clamp_min(0)
        tgt = torch.gather(creatures, 1, safe_slot.unsqueeze(-1).expand(-1, -1, d))
        tgt = tgt * has_tgt.unsqueeze(-1).to(tgt.dtype)

        z_b = z.unsqueeze(1).expand(-1, M, -1)
        oin = torch.cat([kind_e, entity, tgt, z_b], dim=-1)             # [B,M,4d]
        o = F.relu(self.o1(oin))
        o = F.relu(self.o2(o))
        logits = self.o_logit(o).squeeze(-1)                           # [B,M]
        masked_logits = torch.where(opt_mask, logits, logits.new_full((), NEG_INF))

        raw_value = self.v_out(F.relu(self.v1(z))).squeeze(-1)
        value = self.value_scale * torch.tanh(raw_value / self.value_scale)
        return masked_logits, value


def _ff(d: int, mult: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Linear(d * mult, d))


class _PoolLayer(nn.Module):
    """One Perceiver-style pooling block: latents cross-attend to inputs, then self-attend, each
    residual+LayerNorm with a feed-forward."""

    def __init__(self, d: int, heads: int, ff_mult: int):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ff1 = _ff(d, ff_mult)
        self.n2 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff2 = _ff(d, ff_mult)
        self.n4 = nn.LayerNorm(d)

    def forward(self, lat: torch.Tensor, toks: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        a, _ = self.cross(lat, toks, toks, key_padding_mask=key_pad)
        lat = self.n1(lat + a)
        lat = self.n2(lat + self.ff1(lat))
        s, _ = self.self_attn(lat, lat, lat)
        lat = self.n3(lat + s)
        lat = self.n4(lat + self.ff2(lat))
        return lat


def _scatter_add_by_parent(dest: torch.Tensor, src: torch.Tensor, parent: torch.Tensor,
                           n_slots: int) -> torch.Tensor:
    """Add each ``src[b,i]`` token into ``dest[b, parent[b,i]]`` (child -> parent creature fold).

    ``parent`` values are already valid slot indices in ``[0, n_slots)`` (padded rows carry parent 0 but
    a zeroed ``src`` so they add nothing)."""
    B, _, d = dest.shape
    idx = parent.clamp(0, n_slots - 1).unsqueeze(-1).expand(-1, -1, d)
    return dest.scatter_add(1, idx, src)


# ==================================================================================================
# Masked categorical over options (mirrors model_torch).
# ==================================================================================================

def log_prob(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    logp_all = F.log_softmax(logits, dim=-1)
    return logp_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    term = torch.where(torch.isfinite(logp), p * logp, torch.zeros_like(logp))
    return -term.sum(dim=-1)


@torch.no_grad()
def sample_action(logits: torch.Tensor):
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)).clamp_min(1e-20))
    action = (logits + g).argmax(dim=-1)
    return action, log_prob(logits, action)


# ==================================================================================================
# Checkpointing (state_dict + a meta sidecar stamped with the tokenizer parity contract).
# ==================================================================================================

def save_checkpoint(path: str, m: TokenActorCritic) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(m.state_dict(), path)
    meta = {
        "backend": "torch-tokens", "d_model": m.d_model, "n_heads": m.n_heads,
        "n_pool_layers": m.n_pool_layers, "n_latents": m.n_latents, "cat_dim": m.cat_dim,
        "max_options": MAX_OPTIONS, "tokenizer_version": tokens.TOKENIZER_VERSION,
        "tokenizer_signature": tokens.tokenizer_signature(),
        "catalog_signatures": tokens.CATALOG_SIGNATURES,
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)


def load_checkpoint(path: str, device="cpu") -> Tuple[TokenActorCritic, dict]:
    """Load ``(model, meta)``, rejecting a checkpoint whose tokenizer/catalog signature no longer
    matches — exactly like model_torch does with the feature version."""
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    want = tokens.tokenizer_signature()
    if meta.get("tokenizer_signature") != want:
        raise ValueError(
            f"Checkpoint {path} was trained with a different tokenizer/catalog "
            f"(meta={meta.get('tokenizer_signature')} vs current {want}); retrain.")
    m = TokenActorCritic(d_model=meta["d_model"], n_heads=meta["n_heads"],
                         n_pool_layers=meta["n_pool_layers"], n_latents=meta["n_latents"],
                         cat_dim=meta["cat_dim"])
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    return m, meta
