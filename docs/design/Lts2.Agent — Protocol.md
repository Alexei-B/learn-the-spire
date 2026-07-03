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
| `step`  | `index` **or** `cardIndices` | observation (state advanced to the next decision point) |
| `close` | — | `{ "ok": true }`, then the server exits |

A malformed or failed command replies `{ "error": "..." }` **without** stopping the server (so a bad
step index is recoverable). **One run per process** — the game keeps run/combat state in process-wide
singletons; `reset` tears down and restarts the single run. Run N processes for parallelism.

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
