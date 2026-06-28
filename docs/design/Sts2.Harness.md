# Sts2.Harness — Design Reference

Orientation for working on the harness. Read the code for detail; this is the map.

**Status:** vertical slice works — headless boot → map → enter room → full combat
(faithful play + enemy turns) → victory, via the real `sts2.dll`. The public API
(`GetState`/`ListOptions`/`Apply`) now covers the combat + map-move surface. Choice-context
injection and all breadth (rewards/events/shops/bosses/acts 2–3/multiplayer/ascension) are
**not built**; see `docs/plans/`.

## What it is

A C# library that emulates Slay the Spire 2 headlessly by **reusing the real game logic
in `sts2.dll`** — no graphics/audio/engine. Read state, list legal options, apply a
choice, advance deterministically from a seed. Built for fast/parallel sims (future:
agent training).

## Why it's possible

The game is Godot + C# (.NET 9), but the *logic* (`RunState`, `CombatManager`,
`RunManager`, `Player`, `Creature`, `CardModel`, the `Commands.*`/`GameActions.*` layers)
is **plain C#, decoupled from Godot**. UI lives in `Nodes.*` (Godot subclasses) reached
via `N*.Instance` singletons we leave **null** — the logic null-guards them.

## Architecture (3 projects, net9.0, SDK pinned by `global.json`)

- **`src/Sts2.GodotShim`** → builds `GodotSharp.dll` (same assembly name; GodotSharp is
  unsigned so it binds by name). Replaces the real GodotSharp. Two kinds of content:
  pure value types copied verbatim from the decompile (their native calls route to a
  throwing `NativeFuncs`), and inert hand-written facades for engine services
  (`GD`, `OS`, `FileAccess`, `Tween`, node hierarchy, …). **Grown empirically from real
  JIT/load errors.** Node types only need to *exist* + expose members the live logic
  touches; we never instantiate a real game Node, so the source-generator marshalling
  contract is intentionally absent.
- **`src/Sts2.Harness`** → the deliverable. `GameRuntime` (one-time headless boot),
  `HarmonyPatches` (localization degradation), `GameHost` (drives one run).
- **`tests/Sts2.Harness.Tests`** → xUnit; **parallelization disabled** (singletons).
- `refsrc/`, `lib/` are gitignored (decompile + copied game DLLs; GodotSharp excluded).

## Key mechanisms

- **Boot** (`GameRuntime.EnsureInitialized`): mirrors the logic half of the game's
  `OneTimeInitialization`, skipping atlas/UI. Sets `TestMode.IsOn`,
  `NonInteractiveMode` (kills animation/delay/frame waits), mock saves, `ModelDb.Init`,
  etc. See the file for the exact ordered sequence.
- **Localization**: real tables are only in the 1.9 GB `.pck`; Harmony patches make
  missing tables/keys return the key string (mechanics don't need display text).
- **Drive** (`GameHost`): imperative primitives — `StartNewRun`, `EnterFirstRoom`,
  `MoveTo`, `PlayCard` (uses `CardModel.TryManualPlay` = the *faithful* path that pays
  energy; **not** `CardCmd.AutoPlay`, which is free), `EndTurn`. Read via `Run`,
  `Combat`, `InCombat`.
- **Public API** (the read/list/apply trio, built on those primitives):
  `GetState()` → immutable serializable `GameState` DTOs (`GameState.cs`,
  projected by `GameStateProjection`); `ListOptions(playerId)` → `IReadOnlyList<GameOption>`
  (combat card-plays × legal targets + end-turn, map moves, or — when a choice is pending —
  `SelectCards` options); `Apply(GameOption)` → resolves the option and pumps to quiescence.
  `GameOption` carries a serializable description plus internal live references for `Apply`.
- **Choice-context injection** (`HarnessCardSelector`): the game's `CardSelectCmd.Selector`
  seam is replaced with a harness selector. When an effect requests a mid-effect card
  selection (discover/scry/exhaust/search) it calls `GetSelectedCards`, which records a
  `PendingChoice` and blocks the effect's thread-pool task. The harness's combat pump waits
  on whichever comes first — queue drained, or a choice pending — so a blocked choice returns
  control instead of deadlocking; `GetState`/`ListOptions` then surface it (`GamePhase.Choice`)
  and `Apply` resolves it, resuming the effect. Post-combat card-reward selection
  (`GetSelectedCardReward`) is stubbed to pick the first option until rewards land (M2).
- **Async→sync pump**: card plays drain the action queue
  (`ActionExecutor.FinishedExecutingActions`); the enemy turn resolves on fire-and-forget
  tasks, so `EndTurn` waits on a `TaskCompletionSource` wired to combat events
  (`TurnStarted`/`CombatEnded`/`PlayerTurnPhaseChanged`), 5s throwing safety timeout.

## Gotchas

- **One run per process** — run/combat/save state is in process-wide singletons;
  parallelism is multi-process. Reset with `RunManager.CleanUp()`.
- **Faithful vs AutoSlay**: the in-game `AutoSlay` churner plays cards for free; use the
  manual-play path for correctness.
- **The shim is incomplete by design** — each new system surfaces a few more Godot
  members / gated visual branches to stub. Expected cost of breadth.

## Determinism

One master seed → `RunRngSet` derives ~12 named RNG streams. Same seed reproduces the
run. Full snapshot via `RunState.ToSerializable()` ↔ `FromSerializable` (not yet wired in).

## Not built yet

Rewards, events, shops, rest, treasure, elites/bosses, acts 2–3, ascension, local
multiplayer. The `GameState` read model and `ListOptions`/`Apply` span the combat + map-move
surface plus in-combat card-choice injection; still missing: card-reward selection,
enemy-turn-triggered choices, and full multi-select subset enumeration (min &gt; 1 currently
offers a single exact-minimum selection). → `docs/plans/`.
