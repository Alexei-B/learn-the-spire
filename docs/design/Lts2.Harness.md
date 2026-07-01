# Lts2.Harness — Design Reference

Orientation for working on the harness. Read the code for detail; this is the map.

**Status:** vertical slice works — headless boot → **Neow ancient event** → map → enter room →
full combat (faithful play + enemy turns) → victory → **post-combat rewards** → back to the map,
via the real `sts2.dll`. The public API (`GetState`/`ListOptions`/`Apply`) covers the combat +
map-move surface, the battle-rewards screen, **event rooms** (the opening ancient event and the
shared regular-event path), **treasure rooms** (open chest → pick/skip relic), **rest sites**
(rest/smith), **shops** (buy cards/relics/potions, card removal, leave), and **potion use/discard**.
A greedy end-to-end driver (`AutoPlayer` in the tests) plays a run forward through the public API;
buffed to a large HP pool it now plays a **full three-act run start → act-3 boss → the Architect
victory event → win** through the option API (`WalkthroughTests`), ending on `GamePhase.GameOver`
with `GameState.IsVictory`. The act 1→2→3 handoff is driven by reproducing the logic half of the
boss rewards-screen proceed (act transition / victory — see Key mechanisms). Separately,
`Act{1,2,3}FightsTests` / `Act{1,2,3}EventsTests` enumerate *every* fight and event of every act
variant (act 1's Overgrowth + Underdocks — the only index with an alternate — plus the shared events)
and drive each to a terminal state in isolation; **all resolve, no exclusions**, including the two
that needed new harness capability — enemy-turn-triggered player choices (KnowledgeDemon) and the
Trial event's portrait UI (see Key mechanisms). Ascension is plumbed end-to-end and its modifiers
validated (`AscensionTests`; see Ascension below). Remaining breadth (local multiplayer, deck-management
screens, Daily/Custom modes) is **not built**; see `docs/plans/`.

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

## Architecture (5 projects, net9.0, SDK pinned by `global.json`)

- **`src/Lts2.GodotShim`** → builds `GodotSharp.dll` (same assembly name; GodotSharp is
  unsigned so it binds by name). Replaces the real GodotSharp. Two kinds of content:
  pure value types copied verbatim from the decompile (their native calls route to a
  throwing `NativeFuncs`), and inert hand-written facades for engine services
  (`GD`, `OS`, `FileAccess`, `Tween`, node hierarchy, …). **Grown empirically from real
  JIT/load errors.** Node types only need to *exist* + expose members the live logic
  touches; we never instantiate a real game Node, so the source-generator marshalling
  contract is intentionally absent.
- **`src/Lts2.Harness`** → the deliverable. `GameRuntime` (one-time headless boot),
  `HarmonyPatches` (localization degradation), `GameHost` (drives one run).
- **`tests/Lts2.Harness.Tests`** → xUnit; **parallelization disabled** (singletons). Tests drive
  faithful end-to-end flows from a seed; an assembly-wide `BeforeAfterTest` banner
  (`TestLogSeparator`) tags the game's stdout logging with the running test's name. Shared
  navigation (resolve the opening Neow event, move into the first combat) lives in `TestNav`.
- **`src/Lts2.Tui`** → a **full-screen** terminal client (Terminal.Gui **v2** — owns the screen like
  ncurses, soft true-colour theme) that plays full single-player runs purely through the public
  `GetState`/`ListOptions`/`Apply` trio — a manual-testing front end (character/ascension/seed select, then a board canvas,
  a map/piles side panel, a numbered option list, and a scrolling **event log**). It is the only consumer of the
  `StartNewRun(seed, IReadOnlyList<CharacterModel>, ascension)` overload (added for character selection; the
  `playerCount` overloads still assign characters automatically). The event log needs no game-side event
  stream: it **diffs consecutive `GameState`s** (HP/gold/relics/potions/powers, cards gained or moved between
  combat piles, enemy defeats, phase changes). **Save/load** round-trips through the harness's
  `GameHost.ToSaveJson()` / `RestoreFromJson()` (which wrap `Snapshot`/`Restore` over the game's own
  `SerializableRun` JSON) — the app autosaves on reaching the map and offers Continue/Save/Load; snapshotting
  is out-of-combat only. **Event body + per-option outcome text** come from the projection, which renders the
  live event's LocStrings *after binding its dynamic vars* (`Event.DynamicVars.AddTo`, mirroring the game UI) so
  per-run numbers are correct (e.g. "Choose an Attack to Enchant with Sharp 2"). Because a screen-owning driver
  and the game both write to the console, the shim's `GD.Out`/`GD.Err` are **redirectable** (default to the live
  console, so tests are unaffected); the TUI points them at a log file. See `src/Lts2.Tui/README.md`.
- **`src/Lts2.Localization`** → an **opt-in** library that gives real names/descriptions for content
  (cards/relics/potions/powers/events) by reusing the game's own loc pipeline. The game loads its loc
  tables from `res://localization/<lang>/*.json` (plain JSON key→SmartFormat-text dicts) via
  `FileAccess`/`LocManager` — which the shim already backs with real file IO over its globalized
  `res://` root (`<output>/res`), and which `GameRuntime` already initializes. So the library just
  **packages the extracted English tables** as content placed at `res/localization/` (any consumer
  that references it gets them there), and `LocManager.Initialize()` loads them; the Harmony loc
  *finalizers* (which only fire on a missing key) then step aside and real text flows through
  `card.Title` / `card.GetDescriptionForPile()` / etc. `Localizer` exposes id-keyed helpers (the
  read model uses ids) with BBCode stripped, falling back to the id when the tables are absent — so
  the harness/tests, which don't reference this library, stay loc-free (keys), honoring "the headless
  library never requires pak loading". The tables are gitignored game content extracted by
  `scripts/extract-localization.ps1` (GDRE Tools over the `.pck`). See `src/Lts2.Localization`.
- `refsrc/`, `lib/` are gitignored (decompile + copied game DLLs; GodotSharp excluded).

## Key mechanisms

- **Boot** (`GameRuntime.EnsureInitialized`): mirrors the logic half of the game's
  `OneTimeInitialization`, skipping atlas/UI. Sets `TestMode.IsOn`,
  `NonInteractiveMode` (kills animation/delay/frame waits), mock saves (both the settings *and*
  prefs in-memory saves — `PrefsSave` is read by gameplay paths like `TalkCmd.Play`, so an
  uninitialized prefs save silently faults enemy turns), `ModelDb.Init`, etc. See the file for the
  exact ordered sequence.
- **Unlocks**: a run is created with `UnlockState.all` (every epoch unlocked), so all
  content (cards/relics/events) is available and `StartedWithNeow` is true — the run opens on
  the Neow ancient event, like a fully-progressed save. No `SaveManager` epoch override is
  used (which would leak process-wide across runs).
- **Localization**: real tables are only in the 1.9 GB `.pck`; Harmony patches make
  missing tables/keys return the key string (mechanics don't need display text). Event option
  title/description lookups (`EventModel.GetOptionTitle`/`GetOptionDescription` →
  `LocString.GetIfExists`) return *null* for a missing key — not the key — which NRE'd event init
  (`CharacterModel.AddDetailsTo`); a postfix patch degrades those to a key-named `LocString` too.
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
  and `Apply` resolves it, resuming the effect. A choice can also be raised on the **enemy's turn**
  (e.g. KnowledgeDemon's curse selection): the enemy-turn wait (`WaitUntilPlayerCanActOrCombatEnds`)
  wakes on the same effect-suspended signals, so the choice surfaces instead of deadlocking, and
  `Apply(SelectCards)` resumes it via the turn-wait (not the action-queue pump) — distinguished by the
  player's `PlayerTurnPhase` (`None` during the enemy turn vs `Play` for a player-effect choice).
  Post-combat card-reward selection (`GetSelectedCardReward`) returns whichever card the harness staged
  when applying the card reward's `TakeReward` option (see Battle rewards below).
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
  unpaused between combat-end and the next room, so reward effects run. A card reward also surfaces its
  **alternatives** (`CardRewardAlternative.Generate`, projected into `RewardView.CardAlternatives`) as
  `TakeCardRewardAlternative` options: a terminal one (Pael's Wing `SACRIFICE`) is staged by id and
  run through `SelectLocalReward` (the seam resolves it against the live list by id, since the game
  regenerates and reference-matches the alternative each round); a `REROLL` (Driftwood) calls the
  reward's `Reroll()` in place and leaves the screen up with new cards. Plain `Skip` is omitted —
  `ProceedFromRewards` already skips. `GameState.Score` projects `ScoreUtility.CalculateScore`.
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
- **Act transition & victory**: in-game, dismissing an act-boss rewards screen votes to move to the
  next act (`NRewardsScreen` → `ActChangeSynchronizer.SetLocalPlayerReady` → `RunManager.EnterNextAct`).
  The harness reproduces the logic half: `ProceedFromRewards` on a Boss room *reached by travelling to
  the boss map node* (`TryAdvanceActAfterBoss`; a debug-entered encounter or the first of a double-boss
  is excluded) drives `EnterNextAct` directly (`AdvanceToNextAct`), pumped synchronously like
  `EnterFirstRoom` does for `EnterAct` — single-player has no other voters to wait on. A non-final act
  lands on the next act's map; the **final act** enters the `TheArchitect` victory **event room**
  (`IsVictoryRoom`). The Architect event plays like any other (`ChooseEventOption`s advance its
  dialogue); its final option's effect votes to move to the next act, which — now in the victory room —
  *wins the run* (`RunManager.WinRun` → kill all players) on a fire-and-forget chain. `ApplyEventOption`
  detects that option (it bumps the act-floor counter, unlike the dialogue advances) and pumps to the
  terminal game-over (`WaitForGameOver`). A won run is `GamePhase.GameOver` with `GameState.IsVictory`
  (true iff `RunManager.WinTime > 0`, set when the act-3 boss falls), distinguishing it from a death.
- **Treasure rooms** (`GamePhase.Treasure`): entering a `TreasureRoom` runs
  `TreasureRoomRelicSynchronizer.BeginRelicPicking` (its relics), but the chest-open flow and relic
  award are normally driven by the null `NTreasureRoom`/`NTreasureRoomRelicCollection` UI, so the
  harness reproduces their logic halves. On entry (`TryOpenTreasureChest`) it grants the chest gold
  (`DoNormalRewards`) then offers any relic-added extra rewards (`DoExtraRewardsIfNeeded`, which
  goes through `RewardsSet.Offer` → the custom-reward gate, suspended/surfaced like a battle custom
  set). The relics then surface as `TakeTreasureRelic` (one per relic) + `SkipTreasure` options;
  `Apply` of a take calls `PickRelicLocally`, drains the `PickRelicAction`, and consumes the
  resulting `RelicsAwarded` event to `RelicCmd.Obtain` each awarded relic. A singleplayer skip is
  recorded then `OnRoomExited`'d to clear the pending relics so the player is back on the map.
  Leaving a treasure room (after take or skip) is a normal `MoveTo`.
- **Shops** (`GamePhase.Shop`): entering a `MerchantRoom` builds a per-player `MerchantInventory`
  (cards/relics/potions + a card-removal service) in `EnterInternal`; unlike treasure there is no
  synchronizer or null-UI logic-half to reproduce on entry — the inventory just exists. It surfaces
  as `ShopView` (every stocked item with price + affordability); `ListOptions` yields a `BuyShopItem`
  per in-stock, affordable item plus the reachable `MoveTo`s (the shop is left by moving on, like the
  other rooms). `Apply` of a buy runs the entry's faithful `OnTryPurchaseWrapper`, which pays gold and
  grants the item via the same commands as rewards (`CardPileCmd.Add`/`RelicCmd.Obtain`/
  `PotionCmd.TryToProcure`). **Card removal** goes through `OneOffSynchronizer.DoMerchantCardRemoval`,
  which raises a deck card choice on the same `HarnessCardSelector` seam as combat/Smith (surfaced as
  `GamePhase.Choice`, resolved by `Apply(SelectCards)` resuming the suspended purchase task); on
  success the harness calls `entry.SetUsed()` (the logic half of the null `NMerchantCardRemoval`), so
  it is single-use per shop. Relic shop hooks run in the game logic — **The Courier** discounts
  prices (`ModifyMerchantPrice`) and restocks slots after purchase (`ShouldRefillMerchantEntry`).
- **Rest sites** (`GamePhase.RestSite`): entering a `RestSiteRoom` runs
  `RestSiteSynchronizer.BeginRestSite` (its rest/smith options). Unlike treasure there is no UI
  logic-half to reproduce — the action resolves directly through the synchronizer. Each usable
  option (`IsEnabled`) surfaces as a `ChooseRestOption`; `Apply` calls `ChooseLocalOption(index)`,
  whose effect runs as a task pumped to quiescence or a suspended choice. Rest (`HealRestSiteOption`)
  heals 30% max HP and offers any (usually empty) bonus rewards through the custom-reward gate;
  Smith (`SmithRestSiteOption`) raises a deck card choice through the same `HarnessCardSelector` seam
  as combat — surfaced as `GamePhase.Choice` and resolved by `Apply(SelectCards)`, which resumes the
  suspended rest task. A successful action clears the remaining options, leaving the player on the
  map (left via a normal `MoveTo`).
- **Crystal Sphere minigame** (`GamePhase.CrystalSphere`): the most complex event runs an interactive
  grid minigame through a UI screen (`NCrystalSphereScreen`) that is null headless. A Harmony prefix on
  its `ShowScreen` skips the screen and routes the plain-C# `CrystalSphereMinigame` to the host
  (`OnCrystalSphereScreenShown`); the offering event-option task suspends inside `PlayMinigame` on the
  minigame's own completion source. The grid surfaces as `CrystalSphereView` (hidden cells + item
  footprints + revealed flags), with `ClickCrystalSphereCell` (one per hidden cell) and a
  `SetCrystalSphereTool` toggle (Big = 3×3, Small = single). `Apply` drives `CellClicked`; spending the
  last divination completes the minigame, which grants the fully-revealed items' rewards via
  `RewardsCmd.OfferCustom` — the same custom-reward gate as relics/events, so payouts surface as the
  normal reward screen (an item only pays out when its whole footprint is uncovered). The pumps wait on
  a Crystal-Sphere signal too, so a raised minigame returns control instead of deadlocking.
- **Potions** (`UsePotion`/`DiscardPotion` options): a player's belt potions surface in the combat
  Play phase and out of combat on the map/shop (not on the reward/event/treasure/rest/choice screens
  — the game blocks potion use there). `Apply` of a use enqueues a `UsePotionAction` via the faithful
  `PotionModel.EnqueueManualUse` (the path the UI's potion popup drives), then pumps to quiescence (a
  potion raising a card choice surfaces it; one that ends combat triggers its rewards); a targeted
  (AnyEnemy) potion in combat expands to one option per valid enemy, everything else is a single
  untargeted use (Self resolves to the owner). Discard enqueues a `DiscardPotionGameAction`. Usability
  mirrors the game's potion popup: AnyTime usable anywhere, CombatOnly only in combat, None/Automatic
  never manually, gated further by `PassesCustomUsabilityCheck`/`CanRemovePotions`.
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
- **The Godot script-generator attributes must be loadable.** `sts2.dll` carries
  `[assembly: AssemblyHasScripts(Type[])]` (plus per-class `[ScriptPath]`/`[GodotClassName]`) from the
  Godot source generator. When a test run reflects over assembly custom attributes (xUnit does this
  during discovery/reporting), instantiating that attribute resolves *every* listed script type — so
  the shim must define the attribute types **and** every Godot base those scripts derive from, or the
  reflection throws a `TypeLoadException`. That surfaces as xUnit "Catastrophic failure" and a non-zero
  `dotnet test` exit code **even though every test passes** (and it corrupts test-case discovery, so
  cases silently vanish). The shim therefore defines the three generator attributes (`ScriptAttributes.cs`)
  and the script-only node/resource bases the harness never instantiates — UI widgets (`LineEdit`,
  `TextEdit`, `Range`, `AspectRatioContainer`), VFX (`Line2D`, `CpuParticles2D`, `BackBufferCopy`),
  `PathFollow2D`, `RichTextEffect`, `ResourceFormatLoader`, and the `Key` enum (`ScriptOnlyNodes.cs`).
- **`OfferCustom` rewards would auto-take in TestMode**: `RewardsSet.Offer()` (used by relic/event
  custom rewards like Kaleidoscope's bonus cards) **auto-selects every reward** when `TestMode.IsOn`
  if no `RewardsSet.testSelector` is installed. The harness installs one (see Custom rewards under
  Key mechanisms) so the agent chooses instead. The post-combat path is separate — it uses
  `GenerateForRoomEnd` + `BeginRewardsSet`, not `Offer`.

## Determinism, snapshots & restore

One master seed → `RunRngSet` derives ~12 named RNG streams. Same seed reproduces the run bit-for-bit
when replayed from the start (`DeterminismTests` asserts a full state signature matches across two
same-seed runs and diverges across seeds). **Snapshot/restore is wired**: `GameHost.Snapshot()` captures
the game's own `SerializableRun` (`RunManager.ToSave`, with the current room as the save's pre-finished
room); `GameHost.Restore(save, seed)` rebuilds the `RunState` (`RunState.FromSerializable`) and re-enters
it through the logic half of the load path (`SetUpSavedSingleplayer` → `Launch` → `GenerateMap` →
`LoadIntoLatestMapCoord`), minus UI/assets — restoring the full observable state (players, act/floor/
score, map graph). Snapshotting is out-of-combat only (combat state lives in `CombatManager`, not
`RunState`). **Continued play after a restore is not bit-identical** — the engine deliberately does not
persist non-combat RNG across save/load (`MoveToMapCoordAction`: "does not depend on RNGs being
deterministic outside of combat"), so upcoming room-type rolls may differ; this is faithful behavior.

## Ascension

A run is created at an `ascensionLevel` (0–10, a `StartNewRun` param into `RunState`); the game's
`AscensionManager` applies the cumulative per-level modifiers at setup (`InitializeNewRun` →
`ApplyAscensionEffects` for the player-level ones; `GenerateRooms` for the double boss). The level is
projected into the read model as `GameState.AscensionLevel`. `AscensionTests` validates the observable
modifiers through the public state: levels are cumulative (`RunManager.HasAscension`), TightBelt (A4)
shrinks the potion belt, AscendersBane (A5) adds the eternal curse to the starting deck, DoubleBoss
(A10) gives only the final act a distinct `SecondBossEncounter`, and ToughEnemies (A8) raises a
monster's `MinInitialHp`. Every run uses `GameMode.Standard`; Daily/Custom modes are not built.

## Local ("fake") multiplayer

`GameHost.StartNewRun(seed, playerCount, ascension)` builds an N-player run on the **single-process
fake-multiplayer** path: the singleplayer net service (`NetSingleplayerGameService`) hosting N players
(NetIds 1..N, successive `ModelDb.AllCharacters`). The game supports this — `RunManager.
IsSingleplayerOrFakeMultiplayer` is true (net type is Singleplayer regardless of player count), so
turn/choice waits don't block on absent remote peers. The read model already iterates `run.Players`, so
every player's state surfaces; `GetPlayerById(netId)` exposes a live player. **Shared combat**: both
players occupy the same Play phase — each plays their own cards (own deck/hand/energy) via
`ListOptions(netId)`/`Apply`, contributing to the one fight — and the turn ends as a **unit**: under
fake-multiplayer the game treats the round as endable as soon as one player ends
(`CombatManager.AllPlayersReadyToEndTurn` is unconditionally true when `IsSingleplayerOrFakeMultiplayer`),
so `EndTurn` triggers and waits out the enemy turn. `MultiplayerTests` drives a shared 2-player combat to
resolution, asserting both players play cards in the same Play phase. Independent per-player turn-ending
(the enemy waiting for each player to end their own turn) belongs to the real multiplayer net path
(`SetUpNewMultiplayer`), not fake-multiplayer — not built. **Forward navigation
works**: each player resolves their **own** event instance (`ListOptions(netId)` lists that player's
options via `EventSynchronizer.GetEventForPlayer`; `Apply` drives the local player through
`ChooseLocalOption` and any other player through the per-player `ChooseOptionForEvent` seam — reached by
reflection, since the harness originates every player's input and there is no remote client to send the
message the handler would otherwise receive). **Map voting**: `MoveTo(player, coord)` registers a vote
(`MapSelectionSynchronizer.PlayerVotedForMapCoord`); only once every player has voted does the game pick
a destination and move via the faithful `MoveToMapCoordAction` (TestMode → `EnterMapCoord`).
`MultiplayerTests` drives a 2-player run through both players' Neow into a shared first combat. **Shared
(vote-based) events** work for every player: the local player votes via `ChooseLocalOption`, any other
via the per-player `PlayerVotedForSharedOptionIndex` seam (with the live `_pageIndex`); a non-final voter
just records their vote (the event stays actionable only for un-voted players) and the option resolves
for all once everyone has voted. `EventView.Votes` surfaces each player's pending vote (has-voted + which
option) so an agent sees the others' indicated choices before resolution (`MultiplayerTests`). **Treasure
chests** are per-player too: a multi-player chest offers one relic per player; on entry the game
auto-votes the dummies (fake-multiplayer shortcut), which the harness resets (reflecting the
synchronizer's `_votes`) so each player picks for real via `OnPicked` — a non-final picker waits until
everyone has, then the game awards the relics (sole voter gets it; conflicts → rock-paper-scissors).
`TreasureView.Votes` surfaces each player's pending pick. **Rest sites, shops and post-combat rewards** are
per-player too: each player rests on their own slot (`RestSiteSynchronizer.GetOptionsForPlayer` +
`ChooseOption`), buys from their own `MerchantInventory` (`MerchantRoom.Inventories[slot]`) with their own
gold, and receives their own end-of-combat rewards (the game generates one set per alive player; the
harness surfaces them one at a time, routing take/skip by the set owner via `SelectLocalReward`/
`SelectRewardForPlayer`). M6 is **done** for the fake-multiplayer model; the only unmodelled thing is
independent per-player combat turn-ending, which needs the real multiplayer net path. The shared relic
grab-bag stays shared-as-local. See `docs/plans/` M6.

## Property-style E2E fuzzing

`PropertyE2ETests` + `RandomPlayer` fuzz full runs: parameterized by (game seed, input seed), the driver
plays a random legal choice each step through the public option API and checks the mechanical invariants
after every step — HP in `[0, MaxHp]`, non-negative gold/energy, non-empty deck, enemy HP in range, a
non-decreasing floor — and that the run never gets stuck (a non-terminal state with no legal option) or
throws. Two theories over a six-pair corpus: unbuffed (validates the early game + termination) and
HP-buffed (reaches events/treasure/shops/rest/act transitions for breadth). Failing seed pairs are added
to the `SeedPairs` member data to pin regressions. The shim invariant (every native call throws, never an
AccessViolation) is held across the whole suite; a dedicated runs/sec performance pass is not yet done.

The fuzz runs also install a `LogErrorSink` (subscribes to the static `Log.LogCallback`) and fail on any
`Error`-level log — turning **swallowed** exceptions on fire-and-forget tasks (combat/enemy-turn/event
option) into failures, so a faulted effect can't hide behind a healthy-looking run. This caught the
shared `JungleMazeAdventure`/`DenseVegetation` events NRE-ing on null UI singletons inside their
`if (LocalContext.IsMe(owner))` cosmetic blocks (`NDebugAudioManager.Instance` SFX and
`NGame.Instance.ScreenRumble`); the generalized cosmetic-call IL-strip (see below) drops those calls.

## Not built yet

Deck-management screens, multiplayer map navigation / per-player rooms, the Daily/Custom game modes,
a hot-path performance pass.
(Ascension is now plumbed and validated, and multi-player run setup + combat turn sync work — see above.
Custom/alternate reward sets, the run score,
and the full relic roster are now covered — see below. STS2 act bosses have no separate boss-relic
pick. There are no alternate index-1/2 acts — only index 0 has a second variant, Underdocks, which is
covered.) The `GameState` read model and `ListOptions`/`Apply` span the combat + map-move
surface, in-combat *and enemy-turn* card-choice injection, the post-combat battle-rewards screen
(including card-reward alternatives/reroll and the run score), event rooms (opening ancient +
regular-event path), custom relic/event reward sets (`OfferCustom`), treasure rooms, rest sites, shops,
potion use/discard, the act 1→2→3 transition, and the Architect victory.
Still missing: multi-page events and the event `WillKillPlayer` hint; full multi-select subset
enumeration (min &gt; 1 currently offers a single exact-minimum selection). The exotic `OfferCustom`
reward sets and the full relic roster are now covered (see "Relic & reward-set coverage" below).

**Full content sweep (every act variant).** `Act{1,2,3}FightsTests` / `Act{1,2,3}EventsTests`
enumerate *every* fight and event of every act variant (act 1's Overgrowth + Underdocks — the only
index with a second variant — plus the shared events) and drive each to a terminal state through the
public option API, entering each directly via `GameHost.EnterEncounterDebug` / `EnterEventDebug`
(test/dev seams over the game's `EnterRoomDebug`). **All resolve without the harness throwing, with no
exclusions** — including KnowledgeDemon (enemy-turn card choice, surfaced via the enemy-turn wait) and
Trial (event-room portrait UI, neutralized with inert `NEventRoom`/`NEventLayout` stand-ins whose
cosmetic portrait methods are no-op'd; `NEventRoom.Instance` is reached unguarded only by Trial — every
other site uses the null-safe `?.VfxContainer` and nothing branches on it being null). Reaching this
closed several content-specific shim/UI gaps, each on a TestMode-gated or purely-visual path:
- shim value/inert members: `CanvasItem.SetVisible`/`IsVisible`, `Sprite2D.Texture`,
  `GodotObject.Call(StringName, Variant[])`, `Variant(GodotObject)` (act-1 closed
  `GpuParticles2D`/`ParticleProcessMaterial`/`Vector3` and `Node.GetNode`/`GetViewportRect`/…);
- an inert `NAudioManager.Instance` (Harmony-patched getter): a few monsters deref it *unguarded* for
  death SFX, unlike `SfxCmd`'s `NonInteractiveMode` guard; every playback method early-returns on
  `TestMode.IsOn`, so the inert instance makes them no-op;
- the **KaiserCrab** two-part boss (`Crusher`+`Rocket`) reaches into a UI background node that
  *throws* headless — its `Background` getter is patched to one inert `NKaiserCrabBossBackground` and
  that node's `Play*` anim methods are no-op'd;
- the `NGame.Instance.ScreenShakeTrauma` IL-strip transpiler (generalized) covers Amalgamator's combine
  options as well as PunchOff; and SoulNexus's unguarded death-animation handler is no-op'd.
- a cosmetic-call IL-strip (`StripCosmeticUiTranspiler`) drops unguarded calls to null UI singletons from
  event-option state machines — `NDebugAudioManager.Instance.Play/Stop/StopAll` (temporary SFX) and
  `NGame.Instance.ScreenRumble` — for the shared `JungleMazeAdventure` and `DenseVegetation` events,
  whose `if (LocalContext.IsMe(owner))` blocks run headless (the local player *is* "me") and NRE'd the
  option's effect before its mechanical payout. Patching the audio methods directly is impossible
  (Harmony can't read `NDebugAudioManager.Play`'s body — it references `Godot.AudioStream`, absent from
  the shim), so the call is stripped at the call site (pop the null receiver + args, push a default
  return), leaving the gold/heal/finish logic intact (`SharedEventTests`).
→ `docs/plans/`. 

**Relic & reward-set coverage (M4-deferred sweeps).** `RelicSweepTests` grants *every* relic
(`ModelDb.AllRelics`, 294) and drives it through a short seeded combat to post-combat rewards; each
asserts the relic surfaces in `PlayerState.Relics` and the harness never throws/hangs. Granting goes
through **`GameHost.ObtainRelicDebug`** — a fire-and-forget `RelicCmd.Obtain` pumped to quiescence (or a
surfaced choice/reward), because awaiting it inline deadlocks for relics whose `AfterObtained` raises a
custom reward through the harness gate (Orrery/Cauldron/CallingBell/…). The old treasure-only
`_treasureExtraRewardsTask` is generalized to **`_suspendedRoomTask`** (any fire-and-forget room/relic
effect suspended on a custom reward or a card choice); `Apply(SelectCards)` now pumps it too, so a
relic's on-obtain card choice (NewLeaf transform, PreservedFog removal) is awaited to completion instead
of leaving its continuation racing `GetState` (an intermittent null-in-deck projection NRE). One more
loc gap closed: `CardModel.SelectionScreenPrompt` *throws* on a missing key (unlike other lookups), so a
Harmony finalizer degrades it to a key-named `LocString` — letting a card that raises a mid-effect
selection (Wish, added by SereTalon) play instead of faulting. `ExoticRewardSetTests` asserts the shape
of each non-Kaleidoscope `OfferCustom` set (relic-pick / potion / card / mixed), and `RewardSweepTests`
drives every reward screen on a forward run (several seeds) through reroll/take/proceed with invariants.
