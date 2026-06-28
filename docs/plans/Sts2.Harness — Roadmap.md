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
- **Potions**: use (targeted/untargeted), discard.
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
  map. The shared `EventRoom` path covers regular map events too (same projection/seam); still to
  do: events that start combat (shared events) and that raise mid-event card choices, exercised
  end-to-end; multi-page events; the `WillKillPlayer` flag in the projection.
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
- **Shops** (inventory: cards/relics/potions, purchase, card-removal, exit).
- **Rest sites** (rest/smith/and other options).
- **Deck-management screens** (upgrade/transform/enchant/remove/duplicate).

## M4 — Full single-player run (acts + bosses)
- **Map navigation breadth**: full path listing, elites, and end-of-act flow.
- **Act transitions**: act 1 → 2 → 3, including the boss → next-act handoff and rewards.
- **Bosses & elites**: boss encounters and boss relic rewards.
- **Win/lose terminal states**: victory screen, game-over, score.
- Deliverable: a seeded run plays start → act-3 boss with greedy/random legal choices.
- _In progress_: a greedy end-to-end driver (`AutoPlayer` in the tests) plays a run forward through
  events/combats/rewards/map moves via the public option API. On the standard seed it advances
  several act-1 floors and then dies (`WalkthroughTests`) — the "beat the boss or die" loop runs,
  but the greedy combat play is too weak to clear the act and many seeds still surface harness gaps
  (un-handled content → pump timeouts / shim `TypeLoad`s). Reaching the boss needs a stronger combat
  driver and the remaining M3 rooms (rest/shop) so a forward playthrough doesn't stall.

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
