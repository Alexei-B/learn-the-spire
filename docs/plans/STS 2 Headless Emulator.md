# Plan: Headless Slay the Spire 2 emulator library (C#)

## Context

Goal: a C# library that emulates full games of **Slay the Spire 2** by reusing the
real game's logic from `sts2.dll`, exposing three things via a clean API:
1. **Getters** for the complete mechanical game state (map/battle/event, per-player
   status/deck/relics/potions/money, draw/discard/exhaust/hand, counters, relic
   counters, map graph, shop inventory, reward options, etc.).
2. **List available options** in the current state (playable cards per target, map
   moves, shop items + exit, event/ancient choices, reward/relic picks, rest options…).
3. **Take a chosen option**, advancing the game state deterministically.

This becomes the foundation for a future (out-of-scope) RL/agent-training framework, so
it must be **lightweight, deterministic, and parallelizable across processes** — not tied
to the graphical engine.

### Key research findings (drive the whole approach)

- `sts2.dll` is a **managed .NET 9 assembly** (Godot game; ships `sts2.xml` XML docs,
  `sts2.deps.json`). Tooling present: `ilspycmd` 9.1, .NET 9 + 10 SDKs.
- Game logic is **decoupled from Godot**: `RunState`, `CombatState`, `CombatManager`,
  `RunManager`, `Creature`, `CardModel`, `GameAction`, the `Commands.*` layer are plain
  C#. `RunManager.Instance = new RunManager()` and `CombatManager.Instance = new CombatManager()`
  are **plain singletons, not Nodes**.
- Of ~2124 `Nodes.*` types, ~1580 are Screens/Vfx/CommonUi (pure UI to skip). UI access
  inside the managers is **mostly null-guarded** (`NRun.Instance?.`, `NCombatRoom.Instance?.`)
  so it no-ops when UI is absent; a **small set** of paths hard-require `NGame.Instance` /
  `NMapScreen.Instance` (room transitions/fades) and must be neutralized.
- The game is built for automation: `AutoSlay.AutoSlayer` (the devs' own headless auto-player
  with per-room/per-screen handlers), `TestSupport.NonInteractiveMode`, `ICardSelector`,
  `SaveManager.MockInstanceForTesting`, and `RunManager.EnterRoomDebug` / `EnterMapCoordDebug`
  / `DebugOnlyGetState`. **`0Harmony.dll` (Harmony) ships with the game** → we can no-op the
  few hard UI dependencies.
- Driving seam: per-player `GameActions.Multiplayer.ActionQueueSet` + `ActionExecutor`;
  mid-effect decisions go through `GameActions.Multiplayer.PlayerChoiceContext`
  (Blocking / Hook / Throwing) returning `PlayerChoiceResult`. `HookPlayerChoiceContext`
  is the injectable callback point. Commands: `CardCmd`, `PotionCmd`, `PlayerCmd`.
- Run creation: `RunState.CreateForNewRun(players, acts, modifiers, GameMode, ascension, seed)`;
  `Player.CreateForNewRun<TCharacter>(unlockState, netId)`; `RunManager.SetUpNewSingleplayer(...)`.
- Determinism: one master string seed → `RunRngSet` derives ~12 named streams
  (`Random.MegaRandom` = xoshiro256\*\*, `Random.Rng` tracks seed+counter).
  Full snapshot/restore via `Saves.SerializableRun` ↔ `RunState.ToSerializable()` /
  `FromSerializable()`.
- State read surface (confirmed): `RunState` (Players, Acts, Map, CurrentMapCoord,
  CurrentRoom, Gold…), `Player` (Character, Creature, Deck, Relics, PotionSlots, Gold,
  `PlayerCombatState`), `Creature` (CurrentHp/MaxHp/Block/Powers), `CardPile` (Type, Cards),
  `CombatState` (Allies, Enemies, RoundNumber, CurrentSide, Encounter), monster intent via
  `MonsterModel.NextMove.Intents`, `Map.ActMap` / `MapPoint` (coord, PointType, Children/parents),
  `RelicModel` (DisplayAmount/ShowCounter/DynamicVars).

### Decisions (confirmed with user)

- **Architecture: in-process Godot shim** — replace `GodotSharp.dll` with a minimal managed
  shim and Harmony no-op patches; call the managers directly. (Not hosting the real engine.)
- **Scope: vertical slice first, then breadth.**

---

## Decompiled reference (local, gitignored)

- Create `refsrc/` (gitignored). Full ILSpy export of the relevant assemblies for reference
  while building (NOT compiled into the project):
  - `ilspycmd "<game>/sts2.dll" -p -o refsrc/sts2`
  - `ilspycmd "<game>/GodotSharp.dll" -p -o refsrc/GodotSharp` (to know exactly which
    Godot types/members sts2 touches → defines the shim surface).
- `.gitignore`: `refsrc/`, `bin/`, `obj/`, `*.user`, plus a `lib/` holding copies of the
  game DLLs we reference (also gitignored — do not commit copyrighted game binaries).
- Copy the needed game DLLs into `lib/` (sts2.dll, 0Harmony.dll, and the non-Godot deps
  sts2 needs: MonoMod.\*, SmartFormat\*, System.IO.Hashing, JetBrains.Annotations, etc.).
  `GodotSharp.dll` is **not** copied — the shim replaces it.

---

## Solution layout (`learn-the-spire.sln`, net9.0 to match the game)

```
src/
  Lts2.GodotShim/         # managed replacement for GodotSharp (Godot namespace)
  Lts2.Harness/           # the emulator library (public API) — the deliverable
tests/
  Lts2.Harness.Tests/     # xUnit: unit + seeded end-to-end property-style tests
refsrc/  lib/             # gitignored
```

`Lts2.Harness` references `lib/sts2.dll` + game deps + `Lts2.GodotShim` (assembly name
`GodotSharp`, exposing the `Godot` namespace) so sts2's `Godot` references bind to the shim.
Target **net9.0** (sts2 targets net9.0 / references .NET 9 corelib).

---

## Milestone 1 — Vertical slice (definition of done for this iteration's core)

Boot headless → play one full combat → take a card reward → move on the map, all through
the public API, deterministically, with getters and a seeded E2E test.

### 1. `Lts2.GodotShim` — minimal `GodotSharp` replacement
- Build the shim **surface-driven by `refsrc/GodotSharp` + load/runtime errors**, not by
  guessing. Implement just what sts2 binds against:
  - `Godot.GodotObject`, `Node`, `Control`, `RefCounted`, `Resource`, `SceneTree`,
    `StringName`, `NodePath`, math types (`Vector2`, `Vector2I`, `Color`…),
    `Godot.Collections.Array`/`Dictionary`, attributes (`[GlobalClass]`, `[Signal]`,
    `[Export]`, `GodotSourceGenerators` shims), `ToSignal`, `CallDeferred`,
    `ResourceLoader`/`GD` — as **no-op / managed-only** behavior.
  - Node lifecycle (`_Ready`/`_Process`/`_EnterTree`) becomes inert; `GetTree()` returns a
    lightweight stub; signals map to plain C# events; `CallDeferred`/`ToSignal` resolve
    synchronously/immediately (no frames, no delays).
- Acceptance: `sts2.dll` loads against the shim and core logic types (`RunState`,
  `CombatManager`, `CardModel`, …) construct and static-init without throwing.

### 2. `Lts2.Harness` — headless bootstrap (`GameHost`)
- Activate automation flags: `NonInteractiveMode` on, `AutoSlayer.IsActive`/FastMode
  equivalents, `SaveManager.MockInstanceForTesting(...)`, disable FTUE/saving.
- Apply **Harmony patches** to no-op the small set of hard UI/transition/animation
  dependencies (`NGame.Instance.Transition.RoomFadeOut/In`, `NMapScreen.Instance.*`, audio,
  any non-null-guarded `N*.Instance` on the logic path). Patch list grows empirically from
  the first end-to-end boot.
- Provide stub instances only where a non-null `N*.Instance` is genuinely required.
- Create a run without `NGame`: build `Player.CreateForNewRun<Ironclad>(...)`, then
  `RunState.CreateForNewRun(players, acts, modifiers, GameMode.Standard, ascension, seed)`,
  then `RunManager.Instance.SetUpNewSingleplayer(runState, ...)` + `InitializeNewRun` /
  `GenerateMap` / `GenerateRooms`. Use `EnterRoomDebug` / `EnterMapCoordDebug` to advance.
- **Async→sync pump**: the engine is Task-based (`ActionExecutor.ExecuteActions`,
  `FinishedExecutingActions()`). Implement a single-threaded synchronous pump
  (`SynchronizationContext` + drive continuations) so each public "take action" call runs
  the action queue to **quiescence** (no running action, queues empty, waiting on a player
  choice, or combat/room resolved) before returning. This is the central engine-control
  primitive.

### 3. Public API (`Lts2.Harness` — the three deliverables)
- `GameState` getters (read-only projection over live logic objects; no mutation):
  phase (Map/Combat/Event/Shop/Reward/Rest/GameOver), players (status/deck/relics/potions/
  money), combat (hand/draw/discard/exhaust/play, enemies + intents, block/HP/powers,
  counters: cards-drawn, shuffles, relic counters), map graph (nodes/edges/types/current),
  shop inventory, reward & relic-choice options. Backed by the live types from research;
  expose as immutable DTOs so snapshots are stable and serializable.
- `IReadOnlyList<GameOption> ListOptions(playerId)`: enumerate legal actions for the current
  state and player. Combat: playable cards (`CardModel.CanPlay(out reason, out preventer)`)
  × legal targets (`ICombatState.HittableEnemies` / `CardModel.IsValidTarget`), use-potion,
  end-turn. Map: legal `MapCoord` moves. Reward/relic/event/shop/rest: the choice options
  the corresponding screen-context exposes. Each `GameOption` carries the data needed to
  resolve it (action type + card/target/coord/index + owning player).
- `void Apply(GameOption)` / `Apply(playerId, optionIndex)`: resolve via the matching seam —
  `CardCmd`/`PotionCmd`/`PlayerCmd`, `MoveToMapCoordAction`/`VoteForMapCoordAction`,
  `PickRelicAction`, or by satisfying the active `PlayerChoiceContext` with a
  `PlayerChoiceResult` (the Harness installs a `HookPlayerChoiceContext`-style callback that
  blocks on our queued choice). Then pump to quiescence.
- **Choice injection**: replace the player choice context with a Harness-controlled one that,
  when the engine requests a decision, surfaces it through `ListOptions` and waits for the
  next `Apply`. This unifies mid-combat discover/select choices with top-level room choices.

### 4. Tests (`Lts2.Harness.Tests`, xUnit)
- Unit: shim sanity (sts2 loads/constructs), `GameState` getters on a known seeded state,
  `ListOptions` correctness for a constructed combat, RNG determinism (same seed →
  identical derived streams / identical `ToSerializable()`).
- **Seeded end-to-end (property-style)**: parameterized by (input-RNG seed, game seed);
  play the vertical-slice flow with random legal choices; assert no exceptions, invariants
  hold (HP/energy/pile-count sanity, state advances). On failure, persist the seed pair to a
  regression corpus file that is replayed as fixed cases on every run.

---

## Milestone 2+ — Breadth (after the slice proves the architecture)

Widen handler coverage to all room/screen decision points using `AutoSlay.Handlers.*` as the
reference map: events + ancient events, shops (inventory + purchase + remove + exit), rest
sites, treasure, all card-select screens (upgrade/transform/enchant/discover/bundle),
elite/boss, act transitions, full 3-act run to victory/defeat. Add full **local multiplayer**
(N players via per-player `ActionQueueSet` + `PlayerChoiceSynchronizer`; `ListOptions`/`Apply`
already player-scoped). Extend the E2E test to a complete random-choice 3-act playthrough and
grow the seed regression corpus.

Out of scope this iteration: the RL/agent training framework and multi-process orchestration
(the API is being shaped to enable it later).

---

## Verification

- `dotnet build learn-the-spire.sln` (net9.0) succeeds; shim binds, sts2 loads.
- `dotnet test` green: shim/getter/options/RNG unit tests + the seeded E2E slice test.
- Manual smoke: a small console driver boots a run with a fixed seed, prints `GameState`,
  plays a combat via `ListOptions`/`Apply` to a card reward + map move, and a second run with
  the same seed produces an identical `ToSerializable()` trace (determinism check).
- Reproducibility: a saved failing seed pair re-fails before a fix and passes after.

## Primary risks & mitigations
- **Shim fidelity** (biggest): grow it empirically from `refsrc/GodotSharp` + real load
  errors; keep behavior inert/managed-only. Optional later: a real-headless-Godot oracle to
  spot divergence.
- **Hidden hard UI deps** on the logic path: Harmony-patch to no-op (Harmony ships with game).
- **Content loaded from `.pck`/`.tres`**: card/monster/relic *models* are code-defined C#
  classes (`Models.Cards.StrikeIronclad`, …); only art/scenes live in `.pck` and are skipped.
  Verify the model registry populates without `ResourceLoader`; if not, register via reflection.
- **Async→sync correctness**: encapsulate in one well-tested pump; assert quiescence invariants.
```
