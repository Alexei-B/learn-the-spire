# Sts2.Harness — Roadmap to a Full Emulator

High-level remaining work to take the harness from the current vertical slice to a
complete, deterministic Slay the Spire 2 emulator: local multiplayer (multiple agents),
ascension, full 3-act runs, bosses — the whole game. Each milestone is roughly
independently shippable and ordered by dependency. Detail lives in the code and
`docs/design/Sts2.Harness.md`.

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
  `GetSelectedCardReward` seam returns the harness-staged pick). Remaining: enemy-turn-triggered
  player choices, and full multi-select subset enumeration (min &gt; 1 currently offers one
  exact-minimum selection).

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
- **Full act-2 / act-3 content enumeration** — _done_: `Act2FightsTests`/`Act3FightsTests` enumerate
  every Hive (20) and Glory (18) encounter, and `Act2EventsTests`/`Act3EventsTests` every Hive (10)
  and Glory (7) event, each driven to a terminal state through the public option API via the
  `EnterEncounterDebug`/`EnterEventDebug` seams. All resolve except two documented, deferred cases:
  **KnowledgeDemonBoss** (the only monster that raises a *player card choice during the enemy turn* —
  enemy-turn-triggered choices are still un-built) and the **Trial** event (its Accept option drives
  the event through the null `NEventRoom` portrait UI, which cascades through `NEventLayout`). Closing
  these surfaced several content-specific shim/UI gaps (see the design doc): `CanvasItem.SetVisible`/
  `Sprite2D.Texture`/`GodotObject.Call`/`Variant(GodotObject)` in the shim; an inert
  `NAudioManager.Instance` (unguarded death-SFX derefs, all `TestMode`-gated to no-ops); an inert
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
  Remaining for M4 breadth: the **alternate index-1/2 acts** (only the default Hive/Glory are wired
  today — `ActModel.GetDefaultList`), **boss relic / alternate reward** sets, **score**, and the two
  deferred content cases above (enemy-turn choices; the Trial portrait UI).

## M5 — Ascension & game modes
- Plumb `ascensionLevel` end-to-end (already a `StartNewRun` param) and validate the
  ascension modifiers (HP, enemy scaling, double bosses, etc.).
- Confirm `GameMode` variants (Standard; Daily/Custom later) behave.

## M6 — Local multiplayer (multiple agents)
The action/choice model is already per-player (`ActionQueueSet`,
`PlayerChoiceSynchronizer`, `IPlayerCollection`); wire it up for N local players.
- Create runs with multiple `Player`s; `SetUpNewMultiplayer` (or single-process "fake
  multiplayer") path.
- **Per-player `ListOptions`/`Apply`** routed by player id; shared vs per-player choices
  (e.g. map voting `VoteForMapCoordAction`, shared relic grab-bag).
- Synchronize turn structure across players in combat.
- One process still hosts one game; multiple agents drive multiple players in it.

## M7 — Determinism, snapshots & persistence
- Wire `RunState.ToSerializable()` ↔ `FromSerializable` into the harness for
  snapshot/restore and replay.
- Verify a master seed reproduces a full run bit-for-bit (RNG stream coverage).
- Define the save/replay format used by the test corpus.

## M8 — Testing & hardening
- **Seeded property-style E2E**: parameterized by (input-RNG seed, game seed); random
  legal play of full runs; assert invariants (HP/energy/pile sanity, state advances, no
  exceptions). Persist failing seed pairs into a replayed regression corpus.
- **Shim completeness**: keep the value-type copies faithful; ensure every native call
  throws (never AccessViolation).
- **Performance**: measure runs/sec; remove incidental allocations/waits on the hot path.

## Out of scope (future, separate system)
Multi-process orchestration and the RL/agent training framework. The API (read /
list-options / apply, per-player) is being shaped to enable them, but they are not part
of the emulator itself.
