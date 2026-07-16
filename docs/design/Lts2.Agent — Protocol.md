# Lts2.Agent — Cross-process decision protocol

How an external agent (typically Python) exchanges decisions with the C# emulator. It carries the
in-process `IDecisionEngine` seam (`GameState` + legal options in → scored actions out) across a
process boundary, so one policy can both **train against** the harness and **run inside** the TUI.

Companion code: `src/Lts2.Agent` (C#), `python/lts2_agent/` (Python), and the design section
"Cross-process agent interop" in `docs/design/Lts2.Harness.md`.

## Framing & transport

- **Framing:** one JSON object per line ("JSON Lines"), UTF-8, terminated with `\n`. Each message is
  written and flushed as a unit; the peer reads a line and parses it.
- **Encoding:** `System.Text.Json` with the shared `AgentJson.Options` — camelCase property names,
  enums as **strings** (`"Combat"`, `"PlayCard"`), and nulls omitted. Keeping one options instance on
  the C# side guarantees both directions agree on the schema.
- **Transport:** abstracted behind `ILineChannel`. Today the only implementation is `StreamLineChannel`
  over a child process's stdio; a TCP transport can be added later **without changing the message
  schema**. **stdout is reserved strictly for protocol messages on both sides — log to stderr.**
- **Version:** every observation carries `protocolVersion` (currently `1`); bump it on any breaking
  change.

## The shared observation and action encoding

Both protocols use the **same** observation and action encoding — that is what makes a policy portable
between them.

**Observation** (C# → agent):

```json
{
  "protocolVersion": 1,
  "state": { "phase": "Combat", "seed": "...", "floor": 3, "score": 20,
             "players": [ ... ], "combat": { "enemies": [ ... ] }, ... },
  "options": [
    { "kind": "PlayCard", "playerId": 1, "description": "Play Strike -> Goblin",
      "card": { "cardId": "StrikeIronclad", "type": "Attack", "damage": 6, ... },
      "targetCombatId": 42, "handIndex": 1 },
    { "kind": "EndTurn", "playerId": 1, "description": "End Turn" }
  ],
  "done": false,
  "info": { "score": 20, "phase": "Combat", "floor": 3, "act": 0,
            "gameOver": false, "victory": false,
            "players": [ { "currentHp": 78, "maxHp": 80, "gold": 99 } ] }
}
```

- `state` is the immutable `GameState` (`src/Lts2.Harness/GameState.cs`) serialized as-is — the full,
  lossless observation. It already contains everything in `info`.
- `options` is the legal action list, in the exact order of `GameHost.ListOptions()`. Each entry is a
  serialized `GameOption` descriptor (its public getters; the live game refs are internal and never
  serialize). **An action is identified by the entry's index** (its position in this array).
- `done` = the run has ended (`state.isGameOver`; equivalently `options` is empty).
- `info` is a compact block of the scalars a reward function usually wants, so the agent needn't walk
  the whole state. **Reward itself is never computed by C#** — the training loop derives its own.

**Action** (agent → C#):

```json
{ "index": 3 }              // apply options[3]
{ "cardIndices": [0, 2] }   // resolve a "choose N of M" card choice with these card indices
```

- The default form is `index` — the position of a legal option.
- **Multi-select wrinkle:** for a mid-effect card choice where you pick N of M cards
  (`state.phase == "Choice"` with `state.pendingChoice` and `maxSelect > 1`), `options` only enumerates
  the single-pick (and one fixed exact-minimum) shortcuts. To choose any other valid subset, send
  `cardIndices` (indices into `pendingChoice.options`), which routes to `GameHost.ApplyCardChoice`.

## Environment protocol (training)

The external agent is the **driver**; the C# `TrainingEnvironmentServer` (hosted by `Lts2.AgentHost`)
is the environment. Commands (agent → C#), each answered with one observation:

| Command | Fields | Reply |
|---|---|---|
| `reset` | `seed` (default `"AGENT"`), `character` (substring match on a character id; default first), `ascension` (default 0) | observation |
| `reset_combat` | `seed`, `character`, `elitePct` (0.2), `bossPct` (0.05), `act` (0/1/2; default any), `starterDeck` (bool), `deckSpec` (see below); explicit closed-eval form: `cards`+`encounter` (+`relics`, `enemyHp`) | observation (dropped straight into one isolated fight) |
| `step`  | `index` **or** `cardIndices` | observation (state advanced to the next decision point) |
| `close` | — | `{ "ok": true }`, then the server exits |

A malformed or failed command replies `{ "error": "..." }` **without** stopping the server (so a bad
step index is recoverable). **One run per process** — the game keeps run/combat state in process-wide
singletons; `reset` tears down and restarts the single run. Run N processes for parallelism.

### `reset_combat` scenarios and `deckSpec`

`reset_combat` builds an **isolated combat** (random character/relics/encounter, full HP) — the training
regime for the combat policy. Its deck is chosen by the optional `deckSpec` field (single source of truth
for deck construction, all sampling derived deterministically from the fight seed — **same seed + same
spec ⇒ byte-identical deck**):

```json
{"kind": "random",    "cards": 15}                             // random deck from the character's pool
{"kind": "realistic", "removals": [0,3], "additions": [0,3],   // "looks like act 1"
 "weights": {"own": 0.60, "colorless": 0.25, "curse": 0.12, "offCharacter": 0.03}}
{"kind": "explicit",  "cards": ["STRIKE_IRONCLAD", "..."]}     // closed-eval only (needs `encounter`)
```

- **`random`** — `cards` cards drawn uniformly from the character's own pool (the pre-existing behavior).
- **`realistic`** — the character's **starter deck**, minus `N` random cards (`N` uniform in the inclusive
  `removals` range) and plus `M` random cards (`M` uniform in `additions`); each addition draws a pool by
  `weights` (own-character / colorless / curse / off-character) then a card uniformly within it. Added
  cards are **unupgraded** (upgrade realism is a later knob). Only the **starter relic** is granted (no
  random relics), so the built deck is the exact, deterministic set the spec describes. Every field
  defaults exactly as shown, so `{"kind": "realistic"}` alone works.
- **`explicit`** — an exact list of card ids (mirrors the legacy top-level `cards` field); requires
  `encounter`. Collectors must refuse this kind (eval-only).
- **Absent `deckSpec`** = exactly the prior behavior: the character's starter deck when `starterDeck` is
  set, otherwise a random 15-card deck (both plus 5 random relics).

**Status-type cards are never dealt into a deck by any spec.** The observation's `info` block gains the
resolved scenario metadata — `deckSpec` (the resolved kind: `"random"`/`"realistic"`/`"explicit"`/
`"starter"`), and for realistic decks `removedCards` / `addedCards` (card-id lists) — so collectors and
the deck-distribution report can tag outcomes without re-deriving them. `info` already carries
`encounter`/`roomType`/`act`/`won`/`hpLost` for scenario fights.

### Catalog dumps

`Lts2.AgentHost` also has two one-shot catalog dumps (stdout JSON) for building Python-side static feature
tables / embedding indices: `--dump-cards` (per card: `id`, `type`, `rarity`, `pool` title, `category`,
`colorless`/`curse`/`status` flags, `tags`, `keywords`, `varKeys`) and `--dump-powers` (per power: `id`,
`type` Buff/Debuff, `stackType`, `instanceType`, `allowNegative`, `varKeys`).

## Decision protocol (evaluation)

The C# TUI is the driver; the external process is a **policy server**. `ProcessDecisionEngine` sends a
request per auto-play recommendation:

- Request (C# → agent): `{ "type": "evaluate", "protocolVersion": 1, "state": {...}, "options": [...] }`
- Reply (agent → C#): `{ "scores": [ { "index": 3, "score": 9.2, "rationale": "..." }, ... ] }`

`scores` is a (possibly empty) **subset** of the options, each referenced by index. C# maps each back
onto the supplied options → `ScoredOption`. An **empty list means decline** (no recommendation). Any
failure on the C# side — a dead/timed-out process, malformed JSON, or an out-of-range index — is
treated as a decline (logged, never thrown), so a broken agent degrades to "no pick" instead of
crashing the UI.

## Guarantees & non-goals

- **Portability:** a policy that reads an observation and returns an option index works identically in
  both protocols. The Python reference `decision_server` reuses the same policy callable a trainer uses.
- **Determinism:** a given `(seed, character, ascension)` plus a fixed action sequence reproduces a run
  (subject to the harness's post-restore RNG caveat — see the roadmap's M7).
- **Not in scope here:** the learning algorithm, model format, and any multi-process orchestration —
  those live in the training framework that consumes this protocol.
