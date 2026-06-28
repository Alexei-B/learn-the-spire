# Sts2.Harness — Design Reference

Orientation for working on the harness. Read the code for detail; this is the map.

**Status:** vertical slice works — headless boot → **Neow ancient event** → map → enter room →
full combat (faithful play + enemy turns) → victory → **post-combat rewards** → back to the map,
via the real `sts2.dll`. The public API (`GetState`/`ListOptions`/`Apply`) covers the combat +
map-move surface, the battle-rewards screen, and **event rooms** (the opening ancient event and
the shared regular-event path). Remaining breadth (shops/rest/treasure/bosses/acts 2–3/
multiplayer/ascension) is **not built**; see `docs/plans/`.

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
- **`tests/Sts2.Harness.Tests`** → xUnit; **parallelization disabled** (singletons). Tests drive
  faithful end-to-end flows from a seed; an assembly-wide `BeforeAfterTest` banner
  (`TestLogSeparator`) tags the game's stdout logging with the running test's name. Shared
  navigation (resolve the opening Neow event, move into the first combat) lives in `TestNav`.
- `refsrc/`, `lib/` are gitignored (decompile + copied game DLLs; GodotSharp excluded).

## Key mechanisms

- **Boot** (`GameRuntime.EnsureInitialized`): mirrors the logic half of the game's
  `OneTimeInitialization`, skipping atlas/UI. Sets `TestMode.IsOn`,
  `NonInteractiveMode` (kills animation/delay/frame waits), mock saves, `ModelDb.Init`,
  etc. See the file for the exact ordered sequence.
- **Unlocks**: a run is created with `UnlockState.all` (every epoch unlocked), so all
  content (cards/relics/events) is available and `StartedWithNeow` is true — the run opens on
  the Neow ancient event, like a fully-progressed save. No `SaveManager` epoch override is
  used (which would leak process-wide across runs).
- **Localization**: real tables are only in the 1.9 GB `.pck`; Harmony patches make
  missing tables/keys return the key string (mechanics don't need display text).
- **Drive** (`GameHost`): imperative primitives — `StartNewRun`, `EnterFirstRoom`,
  `MoveTo`, `PlayCard` (uses `CardModel.TryManualPlay` = the *faithful* path that pays
  energy; **not** `CardCmd.AutoPlay`, which is free), `EndTurn`. Read via `Run`,
  `Combat`, `InCombat`.
- **Public API** (the read/list/apply trio, built on those primitives):
  `GetState()` → immutable serializable `GameState` DTOs (`GameState.cs`,
  projected by `GameStateProjection`); `ListOptions(playerId)` → `IReadOnlyList<GameOption>`,
  keyed off the current phase: combat card-plays × legal targets + end-turn (`PlayCard`/`EndTurn`),
  map moves (`MoveTo`), event choices (`ChooseEventOption`), reward screens
  (`TakeReward`/`ProceedFromRewards`), or — when a card choice is pending — `SelectCards`;
  `Apply(GameOption)` → resolves the option and pumps to quiescence. `GameOption` carries a
  serializable description plus internal live references for `Apply`.
- **Choice-context injection** (`HarnessCardSelector`): the game's `CardSelectCmd.Selector`
  seam is replaced with a harness selector. When an effect requests a mid-effect card
  selection (discover/scry/exhaust/search) it calls `GetSelectedCards`, which records a
  `PendingChoice` and blocks the effect's thread-pool task. The harness's combat pump waits
  on whichever comes first — queue drained, or a choice pending — so a blocked choice returns
  control instead of deadlocking; `GetState`/`ListOptions` then surface it (`GamePhase.Choice`)
  and `Apply` resolves it, resuming the effect. Post-combat card-reward selection
  (`GetSelectedCardReward`) returns whichever card the harness staged when applying the card
  reward's `TakeReward` option (see Battle rewards below).
- **Battle rewards**: the faithful victory→rewards flow is driven by `NCombatUi.OnCombatWon`,
  which is null headless, so the harness reproduces its logic half. After a won combat fully
  ends (`TryOfferCombatRewards`, run once per `CombatRoom`), it calls
  `RewardsCmd.GenerateForRoomEnd` (populates the `RewardsSet` + reward-modifying hooks, without
  offering) then `RewardsSetSynchronizer.BeginRewardsSet`. The set surfaces as
  `GamePhase.Reward`/`RewardsView`; `ListOptions` yields a `TakeReward` per untaken reward (a
  card reward expands to one option per offered card) plus `ProceedFromRewards`. `Apply` of a
  `TakeReward` calls `RewardsSetSynchronizer.SelectLocalReward` (staging the chosen card on the
  selector first for card rewards); `ProceedFromRewards` calls `SkipLocalRewardsSet` for any
  untaken rewards and returns to the map. All driven on the harness thread; the executor is
  unpaused between combat-end and the next room, so reward effects run.
- **Custom rewards** (relic/event `RewardsCmd.OfferCustom`, e.g. Kaleidoscope's two bonus card
  rewards): these go through `RewardsSet.Offer`, which in `TestMode` auto-takes every reward unless
  a `RewardsSet.testSelector` is installed. The harness installs one (`OnCustomRewardsOffered`):
  it surfaces the set as `GamePhase.Reward` (same `RewardsView`/`TakeReward`/`ProceedFromRewards`
  options as battle rewards) and **blocks the offering effect's task** until the agent resolves —
  the same suspend-and-surface pattern as the card-choice selector. `ProceedFromRewards`
  distinguishes the two: for a custom set it completes the set, unblocks the effect, and pumps it
  to quiescence (the effect then continues — e.g. the Neow option finishes); for a post-combat set
  it just returns to the map. The pumps wait on the custom-reward signal too, so a suspended offer
  returns control instead of deadlocking.
- **Events** (`GamePhase.Event`): an out-of-combat event room (the opening Neow ancient event,
  or a regular map event) surfaces its `EventModel.CurrentOptions` as `ChooseEventOption` options
  (unlocked, non-proceed ones, projected as `EventView`). `Apply` calls
  `EventSynchronizer.ChooseLocalOption(index)`; the option's `Chosen()` effect runs as a
  fire-and-forget task, which the harness awaits via `AwaitPendingOptionTasks` — returning early
  if it blocks on a card choice (same selector seam as combat). `BeginEvent` is itself
  fire-and-forget, so room entry waits (`WaitForEventReady`) for options to be generated. A
  finished event — or one down to only a "proceed" option — is not actionable: the player leaves
  by moving on the map (the in-game proceed drives the null `NMapScreen`, so we model leaving as a
  normal `MoveTo`).
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
- **`OfferCustom` rewards would auto-take in TestMode**: `RewardsSet.Offer()` (used by relic/event
  custom rewards like Kaleidoscope's bonus cards) **auto-selects every reward** when `TestMode.IsOn`
  if no `RewardsSet.testSelector` is installed. The harness installs one (see Custom rewards under
  Key mechanisms) so the agent chooses instead. The post-combat path is separate — it uses
  `GenerateForRoomEnd` + `BeginRewardsSet`, not `Offer`.

## Determinism

One master seed → `RunRngSet` derives ~12 named RNG streams. Same seed reproduces the
run. Full snapshot via `RunState.ToSerializable()` ↔ `FromSerializable` (not yet wired in).

## Not built yet

Shops, rest, treasure, deck-management screens, elites/bosses, acts 2–3, ascension, local
multiplayer. The `GameState` read model and `ListOptions`/`Apply` span the combat + map-move
surface, in-combat card-choice injection, the post-combat battle-rewards screen, and event rooms
(opening ancient + regular-event path), and custom relic/event reward sets (`OfferCustom`). Still
missing: events that start combat or raise mid-event card choices (exercised end-to-end),
multi-page events, the event `WillKillPlayer` hint; card-reward alternatives/reroll;
enemy-turn-triggered choices; and full multi-select subset enumeration (min &gt; 1 currently offers
a single exact-minimum selection). → `docs/plans/`.
