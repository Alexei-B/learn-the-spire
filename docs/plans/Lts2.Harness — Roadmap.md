# Lts2.Harness — Roadmap to a Full Emulator

High-level remaining work to take the harness from the current vertical slice to a
complete, deterministic Slay the Spire 2 emulator: local multiplayer (multiple agents),
ascension, full 3-act runs, bosses — the whole game. Each milestone is roughly
independently shippable and ordered by dependency. Detail lives in the code and
`docs/design/Lts2.Harness.md`.

Conventions for every milestone: grow the GodotSharp shim only as real JIT/load errors
demand; add seeded tests; keep `RunState.ToSerializable()` round-tripping.

---

## M0 — Done (vertical slice)
Headless boot, map gen, move into a room, full combat (faithful play, enemy turns,
victory), reading combat/run state. 7 tests green.

## M1 — Public API surface (in progress)
Turn the imperative `GameHost` primitives into the intended clean interface.
- **`GameState` read model** — _done (combat + map)_: immutable DTOs (`GameState.cs`,
  projected by `GameStateProjection`) covering phase, per-player status/deck/relics/potions/
  gold, combat piles/energy/powers, enemies + intents, and the act's map graph + reachable
  moves. Captured via `GameHost.GetState()`; detached & serializable. Shop/reward/event
  projections wait on those rooms (M2–M3).
- **`ListOptions(playerId)`** — _done (combat + map)_: `GameHost.ListOptions` enumerates
  combat card-plays × legal targets + end-turn, or map moves, as a uniform `GameOption`.
  Potions and screen choices follow with their rooms.
- **`Apply(option)`** — _done_: `GameHost.Apply` resolves the option via the existing
  primitives and pumps to quiescence.
- **Choice-context injection** — _done (in-combat card selections)_: `HarnessCardSelector`
  replaces the game's `CardSelectCmd.Selector`, so mid-effect card selections (discover,
  exhaust, search, scry, …) record a `PendingChoice`, surface through `GetState`
  (`GamePhase.Choice`) / `ListOptions` (`SelectCards` options), and resolve via `Apply`. The
  combat pump waits on queue-drain-or-choice so a blocked effect returns control instead of
  deadlocking. Post-combat card-reward selection now lands with M2's battle rewards (the
  `GetSelectedCardReward` seam returns the harness-staged pick). Multi-select choices
  (`MaxSelect > 1`, e.g. the Regent's CHARGE!!) resolve with any valid subset via
  `ApplyCardChoice(indices)` (the `SelectCards` options still only enumerate single/exact-minimum
  picks). ScrollBoxes' "choose a bundle" selection — a separate seam
  (`CardSelectCmd.FromChooseABundleScreen`) the game auto-resolves under TestMode — is intercepted by a
  Harmony prefix and surfaced as `GamePhase.BundleChoice` / `ChooseBundle` options. Remaining: full
  multi-select subset *enumeration* as discrete options.

## M2 — Combat completeness
- **Battle rewards** — _done_: winning a combat surfaces `GamePhase.Reward` with the room's
  generated `RewardsSet` (gold/potion/relic/card). The harness reproduces the logic half of the
  victory→rewards flow (normally driven by the null `NCombatUi`): on a won fight it calls
  `RewardsCmd.GenerateForRoomEnd` + `RewardsSetSynchronizer.BeginRewardsSet`, then exposes
  `TakeReward` options (card rewards expand to one option per offered card; picking one stages it
  through the `GetSelectedCardReward` seam) and a `ProceedFromRewards` option that skips any
  untaken rewards and returns to the map. Surfaced via `GetState` (`RewardsView`) /
  `ListOptions`, resolved by `Apply`. Relic/event **custom reward sets** (`RewardsCmd.OfferCustom`,
  e.g. Kaleidoscope's bonus card rewards) are also surfaced now, via the `RewardsSet.testSelector`
  seam: the offering effect is suspended until the agent takes/skips and proceeds (otherwise `Offer`
  auto-takes them all in TestMode). Remaining: card-reward alternatives (Skip-as-heal/sacrifice
  relics), reroll, and the more exotic reward sets (Orrery, Calling Bell, …) as encountered.
- **Potions** — _done_: a player's potions surface as `UsePotion`/`DiscardPotion` options in combat
  (Play phase) and out of combat on the map/shop (not on the reward/event/treasure/rest screens,
  where the game blocks potion use). Use runs the faithful manual path (`PotionModel.EnqueueManualUse`,
  the same the UI's potion popup drives); a targeted (AnyEnemy) potion in combat expands to one
  option per valid enemy, everything else is a single untargeted use (Self → owner). Discard enqueues
  a `DiscardPotionGameAction`. Usage gating mirrors the game (AnyTime anywhere; CombatOnly only in
  combat; None/Automatic never manually; plus `PassesCustomUsabilityCheck`/`CanRemovePotions`).
  `PotionTests` cover an AnyTime heal used out of combat, a CombatOnly potion gated out of combat but
  still discardable, discard, and a targeted attack potion thrown at a chosen enemy in combat.
- **Mid-combat player choices**: discover/scry/select-card/choose-enemy effects via the
  injected choice context.
- **Effect coverage sweep**: exercise many cards/relics/powers (each may surface a few
  more gated visual members to stub). Track via a broad combat fuzz test.

## M3 — Non-combat rooms
Drive each room/screen type through `ListOptions`/`Apply`, using `AutoSlay.Handlers.*`
as the reference for what choices exist:
- **Events** + **ancient events** — _in progress_: all epochs are now unlocked at boot
  (`UnlockState.all`), so every run opens on the **Neow ancient event**. Event rooms surface as
  `GamePhase.Event`/`EventView` with one `ChooseEventOption` per unlocked, non-proceed option;
  `Apply` resolves it through the game's `EventSynchronizer.ChooseLocalOption` seam (running the
  option's effect on a thread-pool task, pumped to quiescence or a surfaced card choice). Once an
  event finishes — or is down to only a "proceed" option — the player leaves it by moving on the
  map. The shared `EventRoom` path covers regular map events too (same projection/seam); regular
  events that build their options from text keys (e.g. AromaOfChaos) work now that the harness
  degrades the missing-key option title/description lookups (`GetOptionTitle`/`GetOptionDescription`
  → `LocString.GetIfExists` returned null, NRE'ing in `CharacterModel.AddDetailsTo`) to a key-named
  `LocString` (`AromaOfChaosTests`, which also covers a regular event raising a mid-event card
  choice). **Full act-1 event enumeration**: `Act1EventsTests` now drives *every* act-1 event — both
  index-0 acts (Overgrowth's 13 + Underdocks's 10) plus the 18 shared events — to a terminal state
  (finished → map, or game-over) through the public option API, entering each directly via the new
  `GameHost.EnterEventDebug` seam (built on the game's `EnterRoomDebug`). Several shim gaps surfaced
  and were closed along the way (see M8 / design "Not built yet"). **Crystal Sphere minigame** — _done_:
  the game's most complex event drives an interactive minigame through a UI screen
  (`NCrystalSphereScreen`, null headless); the harness skips that screen (a Harmony prefix on
  `ShowScreen` routes the plain-C# `CrystalSphereMinigame` to the host) and surfaces the fogged grid
  as `GamePhase.CrystalSphere`/`CrystalSphereView` with one `ClickCrystalSphereCell` per hidden cell
  plus a `SetCrystalSphereTool` toggle (Big 3×3 / Small single). Spending the last divination
  completes the minigame, whose revealed-item rewards flow through the existing `OfferCustom`
  custom-reward gate; only items whose whole footprint is uncovered pay out (`CrystalSphereTests`
  asserts full-vs-partial reveal and end-to-end payout). **PunchOff** — _done_: its "Nab" option
  called `NGame.Instance.ScreenShakeTrauma` unguarded — a `callvirt` on a null UI singleton that NRE'd
  *before* the option's relic reward / finish. Rather than make `NGame.Instance` non-null (which would
  defeat the hundreds of `NGame.Instance?.…` guards the logic relies on), a Harmony transpiler strips
  just that cosmetic call from the option's async-state-machine IL (replacing the `callvirt` with
  stack-balancing pops), so the rest of the effect runs (`PunchOffTests` asserts the curse + relic
  payout). **Every act-1 fight and event now resolves through the public option API — no skips.** Still
  to do: per-option event coverage (the driver walks one greedy path); events that start combat
  exercised across every branch; multi-page events; the `WillKillPlayer` flag in the projection.
- **Treasure** (chests/relic pick) — _done_: entering a treasure room surfaces as
  `GamePhase.Treasure`/`TreasureView`. The harness reproduces the logic half of the null
  `NTreasureRoom`/`NTreasureRoomRelicCollection` UI: on entry it opens the chest (grant gold via
  `TreasureRoom.DoNormalRewards`, then `DoExtraRewardsIfNeeded` — relic-added extra rewards route
  through the existing custom-reward gate), and the synchronizer's generated relics surface as
  `TakeTreasureRelic` (one per relic) + `SkipTreasure` options. `Apply` of a take calls
  `PickRelicLocally` and consumes the `RelicsAwarded` event to `RelicCmd.Obtain` the relic (the UI
  normally does this); a skip ends the singleplayer voting so the player returns to the map.
  Tested by jumping the run location straight to the act's real Treasure node (the game's coord
  entry doesn't require adjacency), since playing forward to it isn't reliable yet (see M4).
- **Rest sites** (rest/smith/and other options) — _done_: entering a rest site surfaces as
  `GamePhase.RestSite`/`RestSiteView`. Unlike treasure there is no UI logic-half to reproduce — the
  actions resolve directly through `RestSiteSynchronizer.ChooseLocalOption`: each usable option
  surfaces as a `ChooseRestOption` (Rest heals 30% max HP; Smith raises a deck card choice through
  the same selector seam as combat and upgrades the pick). Disabled options (Smith with no
  upgradable cards) are omitted; after a successful action the game clears the rest, returning the
  player to the map. The Smith card choice resumes its suspended `ChooseLocalOption` task on
  `Apply(SelectCards)`. Tested via the same direct-jump-to-the-map-node approach as treasure.
- **Shops** — _done_: entering a `MerchantRoom` builds the per-player `MerchantInventory` (cards/
  relics/potions + a card-removal service); no synchronizer or UI logic-half runs on entry, so the
  inventory simply exists. It surfaces as `GamePhase.Shop`/`ShopView` (every stocked item with its
  price + affordability), with one `BuyShopItem` option per in-stock, affordable item plus the
  reachable map moves to leave. `Apply` of a buy runs the entry's faithful purchase path
  (`MerchantEntry.OnTryPurchaseWrapper`), which pays gold and grants the item through the same
  commands as rewards (`CardPileCmd.Add`/`RelicCmd.Obtain`/`PotionCmd.TryToProcure`). The
  **card-removal** service raises a deck card choice through the same `HarnessCardSelector` seam as
  combat/Smith (surfaced as `GamePhase.Choice`, resolved by `Apply(SelectCards)`); on success the
  harness marks the entry used (`SetUsed`, the logic half of the null `NMerchantCardRemoval`), so it
  is single-use per shop. Relic hooks that change shop behaviour are exercised by **The Courier**
  (`ShopTests`): its `ModifyMerchantPrice` discount (card-removal 75 → 60) and `ShouldRefillMerchantEntry`
  restock (a bought relic slot refills instead of clearing) both work through the game logic.
- **Deck-management screens** (upgrade/transform/enchant/remove/duplicate).

## M4 — Full single-player run (acts + bosses) — _done (default acts)_
- **Map navigation breadth**: full path listing, elites, and end-of-act flow. _done_
- **Act transitions**: act 1 → 2 → 3, including the boss → next-act handoff. _done_
- **Bosses & elites**: boss encounters across all three acts. _done_
- **Win/lose terminal states**: game-over, and a flagged victory. _done_
- Deliverable: a seeded run plays start → act-3 boss → **win** with greedy legal choices. _done_
- **Act transition / victory** — _done_: dismissing the rewards of an act's boss (a Boss room reached
  by travelling to the boss map node) drives the real `RunManager.EnterNextAct` — the logic half of
  the rewards-screen proceed that the game routes through `ActChangeSynchronizer.SetLocalPlayerReady`
  (`GameHost.TryAdvanceActAfterBoss`/`AdvanceToNextAct`). Non-final acts land on the next act's map;
  the final act enters the **Architect victory event** (`TheArchitect`), whose proceed option votes to
  win the run, killing the players on a fire-and-forget chain that the harness pumps to game-over
  (`WaitForGameOver`). A won run surfaces as `GamePhase.GameOver` with `GameState.IsVictory` true
  (distinguished from a death by `RunManager.WinTime`). `WalkthroughTests` now drives a full three-act
  run **start → act-3 boss → Architect → win** entirely through the public option API.
- **Full content enumeration (all act variants)** — _done_: `Act{1,2,3}FightsTests` /
  `Act{1,2,3}EventsTests` enumerate **every** fight and event of all act variants — act 1's Overgrowth
  *and* Underdocks (the only index with an alternate; index 1/2 have a single act each, Hive/Glory),
  plus the shared events — each driven to a terminal state through the public option API via the
  `EnterEncounterDebug`/`EnterEventDebug` seams. **Every one resolves** (no exclusions). Two cases that
  needed new harness capability:
  - **Enemy-turn-triggered player choices** — _done_: KnowledgeDemon raises a card choice on its *own*
    turn (`ChooseCurse`), which used to deadlock the enemy-turn wait. The wait
    (`WaitUntilPlayerCanActOrCombatEnds`) now also wakes on the effect-suspended signals, so the choice
    surfaces as `GamePhase.Choice`; `Apply(SelectCards)` resumes the enemy turn (via the turn-wait, not
    the action-queue pump, distinguished by `PlayerTurnPhase`).
  - **Trial's event-room portrait UI** — _done_: its Accept option drives `NEventRoom.Instance.Layout.*`
    (unguarded on the null singleton). The harness hands `NEventRoom.Instance`/`.Layout` inert
    stand-ins and no-ops the cosmetic portrait methods, so Accept builds its verdict sub-options.

  Reaching full coverage also closed several content-specific shim/UI gaps (see the design doc):
  `CanvasItem.SetVisible`/`Sprite2D.Texture`/`GodotObject.Call`/`Variant(GodotObject)` in the shim; an
  inert `NAudioManager.Instance` (unguarded death-SFX derefs, all `TestMode`-gated to no-ops); an inert
  KaiserCrab boss background with its cosmetic anim methods no-op'd (the `Crusher`/`Rocket` two-part
  boss reaches into a UI background node that *throws* headless); a generalized
  `NGame.Instance.ScreenShakeTrauma` IL-strip now covering Amalgamator as well as PunchOff; and a
  no-op for SoulNexus's death-animation handler. Two earlier blockers are fixed: the **BygoneEffigy** elite, whose wake move
  stalled the enemy-turn pump because `TalkCmd.Play` NRE'd on a null `SaveManager.Instance.PrefsSave`
  (`GameRuntime` now calls `InitPrefsDataForTest`; `BygoneEffigyTests`); and the **AromaOfChaos**
  event, whose option generation NRE'd in `CharacterModel.AddDetailsTo` because the option text keys
  are missing from our empty loc tables (`GetOptionTitle`/`GetOptionDescription` returned null) — the
  harness now degrades those to a key-named `LocString` (`AromaOfChaosTests`).
  **Full act-1 fight enumeration**: `Act1FightsTests` now drives *every* act-1 encounter — both
  index-0 acts (Overgrowth's 22 + Underdocks's 20, normals/weaks/elites/bosses) — to a terminal
  state (won → map, or game-over) through the public option API, entering each directly via the new
  `GameHost.EnterEncounterDebug` seam. All 42 resolve without the harness throwing; with the player
  buffed to a huge HP pool the greedy `AutoPlayer` wins all but the **Lagavulin Matriarch** boss
  (its SoulSiphon drains Strength/Dexterity each cycle, so the starting deck eventually deals ~0 and
  the long fight ends in a *survivable* game-over — a legitimate loss, not a harness fault). The
  earlier boss blocker is fixed: the **CeremonialBeast** act-1 boss's `Godot.GpuParticles2D`
  `TypeLoadException` is closed (the type and `ParticleProcessMaterial`/`Vector3` it pulls in are now
  in the shim), so all three Overgrowth bosses are now fightable headless.
  Every default-list run plays start → act-3 boss → Architect → win, and every fight/event of every
  act variant resolves.
- **Card-reward alternatives & reroll** — _done_: a post-combat card reward now surfaces its
  alternatives (`CardRewardAlternative.Generate`) alongside the per-card takes, as
  `TakeCardRewardAlternative` options (projected into `RewardView.CardAlternatives`). A terminal
  alternative (Pael's Wing's `SACRIFICE`) is staged by id and run through the rewards synchronizer
  (completing the reward without adding a card); a `REROLL` (Driftwood) re-rolls the offered cards in
  place and is single-use. Plain `Skip` is omitted (ProceedFromRewards already skips). The selector
  seam (`HarnessCardSelector.GetSelectedCardReward`) resolves the staged alternative against the live
  list by id, since the game regenerates and reference-matches it each round (`RewardAlternativesTests`).
- **Run score** — _done_: `GameState.Score` projects `ScoreUtility.CalculateScore` (floors, gold,
  elites/bosses, ascension-scaled), computed with the win flag once the run is a victory
  (`ScoreTests`, and the victory score in `WalkthroughTests`).

### M4 — Deferred: reward-set & relic coverage — _done_
Two reward-screen tasks plus broad relic coverage, deferred from the M4-polish pass above. None blocked
a beaten run; they harden the reward surface and caught content-specific gaps.
- **Exotic custom reward sets — verify & cover** — _done_: `ExoticRewardSetTests` grants each
  non-Kaleidoscope `RewardsCmd.OfferCustom` relic and drives its surfaced `GamePhase.Reward` set to
  completion, asserting the set's *shape*: **CallingBell** (3 relic picks + a curse added), **Cauldron**
  (5 potions → fills the belt), **LostCoffer** (card + potion), **SmallCapsule** (1 relic), and
  **Orrery / GlassEye** (5 card rewards) / **ToyBox** (4 wax relic picks). All route through the
  existing custom-reward gate; no new reward shapes were needed (card / potion / relic / mixed all
  already projected). Granting goes through the new `GameHost.ObtainRelicDebug` seam (see below).
- **General relic coverage sweep** — _done_: `RelicSweepTests` enumerates **every** relic
  (`ModelDb.AllRelics`, 294) and drives each through grant → a short seeded combat → post-combat
  rewards to a terminal state, mirroring `Act{1,2,3}FightsTests` but varying the relic. Each asserts the
  relic surfaces in `PlayerState.Relics` and that the harness never throws/hangs. The sweep grants
  relics the way the game legitimately could in this single-player run: it skips ones the game would
  never grant here (`!IsAllowed(run)`, e.g. the multiplayer-only **MassiveScroll**, whose card pool is
  empty in single-player), pins **SeaGlass**'s character (its Orobas event sets `CharacterId` first),
  and calls the per-player `SetupForPlayer` hook for the event relics that need it
  (**DustyTome/ArchaicTooth/TouchOfOrobas**) — so their on-obtain effects run against real state
  instead of logging context-missing errors. Granting uses the new
  **`GameHost.ObtainRelicDebug`** seam — a fire-and-forget `RelicCmd.Obtain` pumped to quiescence (or a
  surfaced choice/reward), since awaiting it inline deadlocks for relics whose `AfterObtained` raises a
  custom reward through the harness gate (Orrery/Cauldron/CallingBell/…). The sweep closed three gaps:
  - the `_treasureExtraRewardsTask` field was generalized to `_suspendedRoomTask` (a fire-and-forget
    room/relic effect suspended on a custom reward or card choice), and `Apply(SelectCards)` now pumps
    it — so a relic's on-obtain card choice (**NewLeaf** transform, **PreservedFog** removal) is awaited
    to completion instead of leaving the continuation racing `GetState` (an intermittent null-in-deck
    projection NRE);
  - a Harmony finalizer degrades `CardModel.SelectionScreenPrompt` (which *throws* on a missing loc key,
    unlike other lookups) to a key-named `LocString`, so a card raising a mid-effect selection
    (**Wish**, added by **SereTalon**) plays instead of faulting.
- **Reward-screen coverage sweep (property-style)** — _done_: `RewardSweepTests` plays a forward run
  across several seeds and, at every `GamePhase.Reward` screen reached, exercises its option kinds —
  reroll a card reward when offered, take gold, take a card — asserting invariants (gold only grows,
  the deck grows by exactly one per card taken, HP untouched, a reroll leaves the deck unchanged and the
  screen open) before proceeding (skipping any untaken rewards). Folds toward M8's property-style E2E.
- Boss-relic rewards were *not* included: STS2 act bosses give gold/potion/card (the final boss none),
  with no separate boss-relic pick — the original M4 wording was a misnomer.

## M5 — Ascension & game modes — _done (ascension; Standard mode)_
- **Plumb `ascensionLevel` end-to-end** — _done_: the level (already a `StartNewRun` param flowing
  into `RunState`) is now projected into the read model as `GameState.AscensionLevel`, so an agent can
  see the run's difficulty. The game's `AscensionManager` applies the per-level modifiers at run
  setup (`InitializeNewRun` → `ApplyAscensionEffects`; map gen for the double boss).
- **Validate the ascension modifiers** — _done_ (`AscensionTests`): the levels are cumulative
  (`RunManager.HasAscension` true for every level ≤ N, false above), and each observable modifier is
  asserted through the public state — **TightBelt** (A4) shrinks the potion belt by one slot,
  **AscendersBane** (A5) adds a single eternal curse to the starting deck, **DoubleBoss** (A10) gives
  only the final act a distinct `SecondBossEncounter`, and **ToughEnemies** (A8) raises a monster's
  `MinInitialHp` (enemy HP/damage scaling reads the active run's ascension via `AscensionHelper`).
- **`GameMode` variants** — _Standard validated_: every run uses `GameMode.Standard` (the only mode
  the singleplayer boot path sets up). Daily/Custom need their own lobby/daily-seed setup and stay
  deferred (out of M5's scope, as the original note flagged).

## M6 — Local multiplayer (multiple agents) — _done (fake-multiplayer; all rooms per-player)_
The action/choice model is already per-player (`ActionQueueSet`,
`PlayerChoiceSynchronizer`, `IPlayerCollection`); wired up for N local players. Every room an agent can
diverge on is per-player: combat (shared Play phase), per-player + shared/voted events, map voting,
treasure (relic voting), rest, shop, and post-combat rewards — each with vote visibility where the room
is a shared vote. The one thing fake-multiplayer can't model is **independent per-player combat
turn-ending** (the enemy waiting for each player to end their *own* turn); that needs the real
multiplayer net path (`SetUpNewMultiplayer`, `IsMultiplayer()`), out of scope.
- **Create runs with multiple `Player`s** — _done_: `GameHost.StartNewRun(seed, playerCount, ascension)`
  builds an N-player run on the **single-process "fake multiplayer"** path — the singleplayer net
  service hosting N players (NetIds 1..N, successive `ModelDb.AllCharacters`), which the game itself
  supports (`RunManager.IsSingleplayerOrFakeMultiplayer` keeps turn/choice waits from blocking on
  absent remote peers). The read model already iterates `run.Players`, so every player's
  status/deck/relics/potions surface (`MultiplayerTests`). `GetPlayerById(netId)` exposes a player.
- **Shared combat turn structure** — _done_: in the fake-multiplayer model both players occupy the
  *same* Play phase — each plays their **own** cards (own deck/hand/energy) via `ListOptions(netId)`/
  `Apply`, contributing to the shared fight — and the turn ends as a **unit**: the game treats the round
  as endable as soon as one player ends (`CombatManager.AllPlayersReadyToEndTurn` is unconditionally
  true under `IsSingleplayerOrFakeMultiplayer`), so `GameHost.EndTurn` triggers and waits out the enemy
  turn. `MultiplayerTests` drives a shared 2-player combat to resolution, asserting both players get to
  play cards in the same Play phase. **Note:** independent per-player turn-ending (the enemy waiting for
  every player to end their *own* turn) is a property of the real multiplayer net path
  (`SetUpNewMultiplayer`, `IsMultiplayer()`), **not** fake-multiplayer — out of current scope.
- **Per-player events + map voting (forward navigation)** — _done_: each player resolves their **own**
  event instance — `ListOptions(netId)` lists that player's `ChooseEventOption`s (via
  `EventSynchronizer.GetEventForPlayer`); `Apply` routes the local player through `ChooseLocalOption`
  and any other player through the per-player `ChooseOptionForEvent` seam (the path the game's
  net-message handler would invoke — reached by reflection, since the harness is the input source for
  every player and there is no remote client to send the message). A player who finished their own
  event waits (no options) until the party moves. **Map voting**: `MoveTo(player, coord)` registers the
  player's vote (`MapSelectionSynchronizer.PlayerVotedForMapCoord`); only once *every* player has voted
  does the game pick a destination and move, driven by the faithful `MoveToMapCoordAction` (TestMode →
  `EnterMapCoord`). `MultiplayerTests` drives a 2-player run: each resolves their own Neow, then the
  party votes together into the first combat.
- **Shared (vote-based) events** — _done_: every player votes on a shared event through the same option
  API — the local player via `ChooseLocalOption`, any other player via the per-player
  `PlayerVotedForSharedOptionIndex` seam (with the live `_pageIndex`). A non-final voter just records
  their vote (the event stays actionable only for players who have not voted); once everyone has voted
  the game picks an option and resolves it for all. **Vote visibility**: `EventView.Votes` surfaces each
  player's pending vote (whether they have voted, and for which option) so an agent can see the others'
  indicated choices before resolution (`MultiplayerTests` drives a 2-player shared event end to end).
- **Treasure chests (per-player relic voting)** — _done_: a multi-player chest generates one relic per
  player; on entry the game *auto-votes* the dummies (the fake-multiplayer shortcut), which the harness
  resets so every player picks for real through the option API. Each player takes a relic (or skips) via
  `TreasureRoomRelicSynchronizer.OnPicked`; a non-final picker waits (no options) until everyone has
  picked, when the game awards the relics (a player who alone voted a relic gets it; conflicts go to
  rock-paper-scissors). **Vote visibility**: `TreasureView.Votes` surfaces each player's pending pick
  (`TreasureRoomRelicSynchronizer.GetPlayerVote`). `MultiplayerTests` drives a 2-player chest: each picks
  a different relic and gets it, seeing the other's indicated pick mid-vote.
- **Rest sites, shops, post-combat rewards** — _done_: each player acts on their **own** slot.
  **Rest**: `ListOptions(netId)` lists that player's options (`RestSiteSynchronizer.GetOptionsForPlayer`);
  the local player rests via `ChooseLocalOption`, others via the per-player `ChooseOption` seam — a
  player who has rested waits until everyone is done. **Shop**: each player buys from their own
  `MerchantInventory` (`MerchantRoom.Inventories[slot]`) with their own gold. **Post-combat rewards**:
  the game gives each alive player their own set (`CombatRoom.OfferRoomEndRewards` loops the players);
  the harness generates one per player and surfaces them one at a time, routing take/skip by the set's
  owner (`SelectLocalReward`/`SkipLocalRewardsSet` for the local player, the per-player
  `SelectRewardForPlayer`/`SkipRewardsSetOnStackTopForPlayer` seams for the others). `MultiplayerTests`
  covers all three.
- One process still hosts one game; multiple agents drive multiple players in it.

## M7 — Determinism, snapshots & persistence — _done (snapshot/restore + determinism)_
- **Snapshot/restore** — _done_: `GameHost.Snapshot()` captures the game's own `SerializableRun` save
  model (`RunManager.ToSave`, recording the current room as the save's pre-finished room);
  `GameHost.Restore(save, seed)` rebuilds the `RunState` (`RunState.FromSerializable`) and re-enters it
  through the logic half of the game's load path (`SetUpSavedSingleplayer` → `Launch` → `GenerateMap`
  → `LoadIntoLatestMapCoord`), minus UI/assets. `DeterminismTests` snapshots on the map after a combat,
  restores into a fresh run, and asserts the restored observable state (players, act/floor/score, map
  graph) matches the snapshot exactly, then plays the restored run forward. Snapshotting mid-combat is
  not supported (combat state lives in `CombatManager`, not `RunState`).
- **A master seed reproduces a full run bit-for-bit** — _done_: `DeterminismTests` plays two
  independent runs of the same seed forward through identical greedy inputs and asserts a full state
  signature (players/deck/relics, map graph, combat roster) matches exactly, and that a different seed
  diverges. **Caveat (faithful):** continued play *after a restore* is not bit-identical — the engine
  deliberately does not persist non-combat RNG across save/load (`MoveToMapCoordAction`: "does not
  depend on RNGs being deterministic outside of combat"), so the upcoming room-type rolls may differ.
  Combat and from-the-start replay are deterministic.
- **Save/replay format** — the harness reuses the game's `SerializableRun`; the input-driven replay
  corpus is folded into M8's property-style E2E.

## M8 — Testing & hardening — _done (property-style E2E)_
- **Seeded property-style E2E** — _done_: `PropertyE2ETests` is parameterized by (game seed, input
  seed). `RandomPlayer` plays a full run forward making a *random* legal choice each step (its own
  input RNG, independent of the game seed), through the public option API, asserting the mechanical
  invariants after **every** step — HP within `[0, MaxHp]`, non-negative gold/energy, a non-empty deck,
  enemies' HP in range, a non-decreasing floor — and that the run never gets **stuck** (a non-terminal
  state the harness offers no legal option for) or throws. Two theories over the same six-pair corpus:
  one unbuffed (dies fast, validates the early game + termination) and one HP-buffed (survives into
  events/treasure/shops/rest/act transitions for breadth). A failing (game-seed, input-seed) pair is
  named in the case data, pinning the regression for replay — the corpus is the `SeedPairs` member data,
  grow it with any pair that surfaces a fault.
- **Shim completeness** — _maintained_: the value-type copies are faithful to `refsrc/GodotSharp` and
  every `NativeFuncs.godotsharp_*` routes to a throwing stub (a clean managed exception, never an
  AccessViolation); the shim is grown only as real JIT/load errors demand. No regressions across the
  full suite (511 tests) or the fuzz corpus.
- **Performance** — _not pursued_: no hot-path optimization pass yet (runs/sec measurement, allocation
  trimming). The fuzz/determinism suites run fast enough for CI; a dedicated throughput pass is left for
  when agent-training throughput demands it (out of the current scope).

## Out of scope (future, separate system)
Multi-process orchestration and the RL/agent training framework. The API (read /
list-options / apply, per-player) is being shaped to enable them, but they are not part
of the emulator itself.
