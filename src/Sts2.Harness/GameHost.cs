using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace Sts2.Harness;

/// <summary>
/// Owns a single headless Slay the Spire 2 run: boots the game's logic singletons
/// without the Godot UI, then drives them deterministically from a seed.
/// </summary>
public sealed class GameHost
{
    /// <summary>The live run state. Treat as read-only from outside the harness.</summary>
    public RunState Run { get; private init; } = null!;

    /// <summary>The seed this run was created with.</summary>
    public string Seed { get; private init; } = "";

    /// <summary>
    /// The harness-controlled card selector installed as the game's <c>CardSelectCmd.Selector</c>.
    /// Surfaces mid-effect card choices so they resolve through <see cref="ListOptions(ulong)"/>
    /// / <see cref="Apply"/> instead of a UI screen.
    /// </summary>
    internal HarnessCardSelector Selector { get; private init; } = null!;

    // The CardSelectCmd selector stack is process-wide; track our scope so a new run can
    // replace the previous run's selector cleanly.
    private static IDisposable? _selectorScope;

    /// <summary>
    /// The post-combat rewards currently on offer, or null when not on the rewards screen.
    /// Surfaced as <see cref="GamePhase.Reward"/> and resolved through the reward options.
    /// </summary>
    internal MegaCrit.Sts2.Core.Rewards.RewardsSet? PendingRewards { get; private set; }

    // The combat room whose rewards we have already generated, so we offer them exactly once
    // per fight even while the player lingers on the rewards screen before moving on.
    private MegaCrit.Sts2.Core.Rooms.AbstractRoom? _rewardedRoom;

    // The treasure room whose chest we have already opened, so the gold/extra-reward flow runs
    // exactly once per room (the relic picking that follows is driven by the agent's options).
    private MegaCrit.Sts2.Core.Rooms.TreasureRoom? _openedTreasureRoom;

    // A treasure room's "extra rewards" (DoExtraRewardsIfNeeded) task, kept in flight when it
    // suspends on a custom reward screen so ProceedFromRewards can resume it. Null otherwise.
    private System.Threading.Tasks.Task? _treasureExtraRewardsTask;

    // A rest-site option resolution (ChooseLocalOption) task, kept in flight when it suspends on a
    // mid-effect card choice (e.g. Smith's upgrade selection) so Apply can resume it. Null otherwise.
    private System.Threading.Tasks.Task? _restChoiceTask;

    // Custom rewards offered mid-effect by a relic/event (RewardsCmd.OfferCustom, e.g.
    // Kaleidoscope's two bonus card rewards). Unlike a post-combat set, the offering effect's
    // task is *suspended* inside RewardsSet.Offer until the agent takes/skips and proceeds, so
    // the agent makes the same explicit take-or-skip choice rather than the rewards being
    // silently auto-taken (which is what RewardsSet.Offer does in TestMode without a selector).
    private readonly object _rewardGate = new();
    private System.Threading.Tasks.TaskCompletionSource? _customRewardResolve;
    private System.Threading.Tasks.TaskCompletionSource _customRewardSignal = NewSignal();

    private static System.Threading.Tasks.TaskCompletionSource NewSignal() =>
        new(System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);

    private System.Threading.Tasks.Task CustomRewardSignal
    {
        get { lock (_rewardGate) { return _customRewardSignal.Task; } }
    }

    // The Crystal Sphere event minigame currently awaiting the agent's cell-clicks, or null. The
    // game drives this through a UI screen (NCrystalSphereScreen) that is null headless; a Harmony
    // patch on NCrystalSphereScreen.ShowScreen routes the minigame here instead (see
    // OnCrystalSphereScreenShown) and the offering event-option task suspends inside PlayMinigame on
    // its own completion source until the agent spends every divination. The minigame logic itself
    // (CrystalSphereMinigame) is plain C#; only the screen was UI.
    private readonly object _crystalGate = new();
    private MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? _pendingCrystalSphere;
    private System.Threading.Tasks.TaskCompletionSource _crystalSphereSignal = NewSignal();

    // Process-wide hook the ShowScreen Harmony patch calls; the latest run owns it (like testSelector).
    internal static System.Action<MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame>?
        CrystalSphereScreenHook;

    private System.Threading.Tasks.Task CrystalSphereSignal
    {
        get { lock (_crystalGate) { return _crystalSphereSignal.Task; } }
    }

    /// <summary>The Crystal Sphere minigame awaiting cell-clicks, or null. Read by the projection.</summary>
    internal MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? PendingCrystalSphere
    {
        get { lock (_crystalGate) { return _pendingCrystalSphere; } }
    }

    /// <summary>True while a Crystal Sphere minigame is awaiting the agent's divinations.</summary>
    private bool CrystalSpherePending
    {
        get { lock (_crystalGate) { return _pendingCrystalSphere is not null; } }
    }

    /// <summary>True while a custom (blocking) reward set is awaiting the agent's resolution.</summary>
    private bool CustomRewardPending
    {
        get { lock (_rewardGate) { return _customRewardResolve is not null; } }
    }

    /// <summary>
    /// True when an effect is suspended waiting on the agent — either a mid-effect card choice or
    /// a custom rewards set. The combat/event pumps return control (rather than draining) so the
    /// suspended decision surfaces instead of deadlocking.
    /// </summary>
    private bool EffectSuspended => Selector.Pending is not null || CustomRewardPending || CrystalSpherePending;

    /// <summary>
    /// Create and start a fresh single-player run with the given seed.
    /// Mirrors the game's own NSceneBootstrapper flow, minus all UI/asset steps.
    /// </summary>
    public static GameHost StartNewRun(string seed, int ascension = 0)
    {
        GameRuntime.EnsureInitialized();

        // The game keeps run/combat state in process-wide singletons, so only one run
        // can exist per process. Tear down any previous run before starting a new one.
        if (RunManager.Instance.IsInProgress)
        {
            RunManager.Instance.CleanUp();
        }

        // The harness plays the full game: treat every epoch as unlocked so all content
        // (cards/relics/events) is available and the run starts with the Neow ancient event
        // (StartedWithNeow is derived from the run's unlock state — see RunManager).
        UnlockState unlock = UnlockState.all;

        CharacterModel character = ModelDb.AllCharacters.First(); // Ironclad
        Player player = Player.CreateForNewRun(character, unlock, 1uL);

        List<ActModel> acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();
        RunState runState = RunState.CreateForNewRun(
            new List<Player> { player },
            acts,
            new List<ModifierModel>(),
            GameMode.Standard,
            ascension,
            seed);

        RunManager.Instance.SetUpNewSingleplayer(runState, shouldSave: false);
        RunManager.Instance.Launch();

        // Install our card selector so mid-effect choices route to the harness. Disposing the
        // previous scope clears the process-wide selector stack before we claim it.
        _selectorScope?.Dispose();
        var selector = new HarnessCardSelector();
        _selectorScope = MegaCrit.Sts2.Core.Commands.CardSelectCmd.UseSelector(selector);

        var host = new GameHost { Run = runState, Seed = seed, Selector = selector };

        // Intercept relic/event custom reward sets (RewardsSet.Offer): without a selector the
        // game auto-takes them all in TestMode. Route them through the harness so the agent
        // chooses. The hook is process-wide; the latest run owns it.
        MegaCrit.Sts2.Core.Rewards.RewardsSet.testSelector = host.OnCustomRewardsOffered;

        // Intercept the Crystal Sphere event minigame's UI screen (null headless) so its grid
        // surfaces as agent choices instead. Process-wide; the latest run owns it.
        CrystalSphereScreenHook = host.OnCrystalSphereScreenShown;

        return host;
    }

    /// <summary>
    /// The seam invoked (via a Harmony patch on <c>NCrystalSphereScreen.ShowScreen</c>) when the
    /// Crystal Sphere event would open its UI minigame. Records the live <c>CrystalSphereMinigame</c>
    /// so it surfaces as <see cref="GamePhase.CrystalSphere"/>; the offering event-option task then
    /// suspends inside <c>PlayMinigame</c> on the minigame's completion source until the agent spends
    /// every divination (see <see cref="ClickCrystalSphereCell"/>). Runs on that option task's thread.
    /// </summary>
    internal void OnCrystalSphereScreenShown(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame grid)
    {
        lock (_crystalGate)
        {
            if (_pendingCrystalSphere is not null)
            {
                throw new InvalidOperationException("A Crystal Sphere minigame is already pending.");
            }
            _pendingCrystalSphere = grid;
            _crystalSphereSignal.TrySetResult();
        }
    }

    /// <summary>
    /// The game's <c>RewardsSet.testSelector</c> seam: invoked by <c>RewardsSet.Offer</c> (used by
    /// relic/event custom rewards) on the offering effect's thread-pool task. Surfaces the set as
    /// <see cref="GamePhase.Reward"/> and blocks that task until the agent takes/skips and proceeds,
    /// at which point <see cref="ProceedFromRewards"/> completes the returned task so the effect
    /// resumes. Mirrors the card-choice injection (<see cref="HarnessCardSelector"/>).
    /// </summary>
    internal System.Threading.Tasks.Task OnCustomRewardsOffered(MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
        System.Threading.Tasks.TaskCompletionSource resolve = NewSignal();
        lock (_rewardGate)
        {
            if (PendingRewards is not null)
            {
                throw new InvalidOperationException(
                    "A rewards set is already pending when a custom reward was offered.");
            }
            PendingRewards = set;
            _customRewardResolve = resolve;
            _customRewardSignal.TrySetResult();
        }
        return resolve.Task;
    }

    /// <summary>
    /// Advance from a freshly-launched run into the first act's opening room
    /// (mirrors NSceneBootstrapper's Unassigned/EnterAct path). Async game work is
    /// pumped to completion synchronously.
    /// </summary>
    public void EnterFirstRoom()
    {
        RunManager rm = RunManager.Instance;
        Pump(rm.SetActInternal(0));
        rm.RunLocationTargetedBuffer.OnLocationChanged(Run.RunLocation);
        rm.MapSelectionSynchronizer.OnLocationChanged(Run.MapLocation);
        Pump(rm.EnterAct(0));
        // With all epochs unlocked the run opens on the Neow ancient event; wait for its
        // options to be generated (BeginEvent runs as a fire-and-forget task).
        WaitForEventReady();
    }

    /// <summary>The current combat, or null if not in a battle.</summary>
    public MegaCrit.Sts2.Core.Combat.CombatState? Combat =>
        RunManager.Instance.IsInProgress
            ? MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState()
            : null;

    /// <summary>True while a battle is in progress.</summary>
    public bool InCombat => MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress;

    /// <summary>
    /// Move to a reachable map coordinate, entering its room. In headless mode this
    /// uses the logic path (RunManager.EnterMapCoord), not the map-screen UI.
    /// </summary>
    public void MoveTo(MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        Pump(RunManager.Instance.EnterMapCoord(coord));
        // If we entered an event room, wait for its options to be generated before returning.
        WaitForEventReady();
        // If we entered a treasure room, open the chest (grant gold + offer any extra rewards) so
        // its relics surface as a choice.
        TryOpenTreasureChest();
    }

    /// <summary>
    /// Test/dev seam: enter a combat room for an arbitrary encounter directly, bypassing map
    /// navigation, so every act fight can be exercised in isolation. Mirrors the logic the game's
    /// own <c>RunManager.EnterMapPointInternal</c> runs for a combat point (pause the action
    /// executor, then enter the room via the sanctioned <c>EnterRoomDebug</c> test path). The
    /// encounter is cloned to a fresh mutable instance; returns once combat is set up and ready.
    /// </summary>
    public MegaCrit.Sts2.Core.Rooms.CombatRoom EnterEncounterDebug(
        MegaCrit.Sts2.Core.Models.EncounterModel encounter)
    {
        MegaCrit.Sts2.Core.Models.EncounterModel mutable =
            encounter.IsMutable ? encounter : encounter.ToMutable();
        RunManager.Instance.ActionExecutor.Pause();
        var room = (MegaCrit.Sts2.Core.Rooms.CombatRoom)Pump(
            RunManager.Instance.EnterRoomDebug(
                mutable.RoomType, MegaCrit.Sts2.Core.Map.MapPointType.Unassigned, mutable, showTransition: false));
        DrainActionQueue();
        return room;
    }

    /// <summary>
    /// Test/dev seam: enter an event room for an arbitrary event directly, bypassing map
    /// navigation, so every act event can be exercised in isolation. Mirrors the harness's normal
    /// room-entry handling (wait for the event's options to be generated). The event is cloned to a
    /// fresh mutable instance.
    /// </summary>
    public MegaCrit.Sts2.Core.Rooms.EventRoom EnterEventDebug(
        MegaCrit.Sts2.Core.Models.EventModel ev)
    {
        // Unlike a CombatRoom (which needs a mutable encounter), an EventRoom takes the *canonical*
        // event and makes its own per-player mutable copy, so pass the canonical model through.
        MegaCrit.Sts2.Core.Models.EventModel canonical =
            ev.IsCanonical ? ev : MegaCrit.Sts2.Core.Models.ModelDb.GetById<MegaCrit.Sts2.Core.Models.EventModel>(ev.Id);
        RunManager.Instance.ActionExecutor.Pause();
        var room = (MegaCrit.Sts2.Core.Rooms.EventRoom)Pump(
            RunManager.Instance.EnterRoomDebug(
                MegaCrit.Sts2.Core.Rooms.RoomType.Event, MegaCrit.Sts2.Core.Map.MapPointType.Unassigned, canonical, showTransition: false));
        WaitForEventReady();
        return room;
    }

    /// <summary>
    /// The local player's mutable event when the current room is an (out-of-combat) event room,
    /// or null otherwise. Used to surface <see cref="GamePhase.Event"/> options.
    /// </summary>
    internal MegaCrit.Sts2.Core.Models.EventModel? CurrentEvent =>
        RunManager.Instance.IsInProgress
        && !InCombat
        && Run.CurrentRoom is MegaCrit.Sts2.Core.Rooms.EventRoom
            ? RunManager.Instance.EventSynchronizer.GetLocalEvent()
            : null;

    /// <summary>
    /// True when the current event is awaiting a real (non-proceed, unlocked) choice from the
    /// player. A finished event — or one whose only remaining option is "proceed" — is not
    /// actionable: the player leaves it by moving on the map.
    /// </summary>
    internal bool HasActionableEvent =>
        CurrentEvent is { } e && e.CurrentOptions.Any(o => !o.IsLocked && !o.IsProceed);

    /// <summary>The current room as a treasure room, or null when not standing in one.</summary>
    internal MegaCrit.Sts2.Core.Rooms.TreasureRoom? CurrentTreasureRoom =>
        RunManager.Instance.IsInProgress && !InCombat
            ? Run.CurrentRoom as MegaCrit.Sts2.Core.Rooms.TreasureRoom
            : null;

    private MegaCrit.Sts2.Core.Multiplayer.Game.TreasureRoomRelicSynchronizer TreasureSync =>
        RunManager.Instance.TreasureRoomRelicSynchronizer;

    /// <summary>
    /// True while standing in a treasure room whose chest is open and which still has relics to
    /// pick. Surfaces as <see cref="GamePhase.Treasure"/> with take/skip options. A skipped or
    /// emptied chest leaves no relics, so the player is back on the map.
    /// </summary>
    internal bool HasTreasureChoice =>
        CurrentTreasureRoom is not null
        && PendingRewards is null
        && Selector.Pending is null
        && TreasureSync.CurrentRelics is { Count: > 0 };

    /// <summary>The current room as a rest site, or null when not standing in one.</summary>
    internal MegaCrit.Sts2.Core.Rooms.RestSiteRoom? CurrentRestSiteRoom =>
        RunManager.Instance.IsInProgress && !InCombat
            ? Run.CurrentRoom as MegaCrit.Sts2.Core.Rooms.RestSiteRoom
            : null;

    private MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer RestSiteSync =>
        RunManager.Instance.RestSiteSynchronizer;

    /// <summary>
    /// True while at a rest site that still has at least one usable action. Choosing one consumes
    /// the rest action (the game clears the remaining options), after which the player is back on
    /// the map.
    /// </summary>
    internal bool HasRestChoice =>
        CurrentRestSiteRoom is not null
        && PendingRewards is null
        && Selector.Pending is null
        && RestSiteSync.GetLocalOptions().Any(o => o.IsEnabled);

    /// <summary>
    /// Play a card from hand at an optional target via the canonical player path
    /// (CardModel.TryManualPlay), which validates targeting, spends energy, and runs
    /// the card's effects, then drains the action queue. Returns false if the play was
    /// rejected (e.g. not enough energy / invalid target). Pass the target for
    /// AnyEnemy/AnyAlly cards; null otherwise.
    /// </summary>
    public bool PlayCard(MegaCrit.Sts2.Core.Models.CardModel card, MegaCrit.Sts2.Core.Entities.Creatures.Creature? target)
    {
        bool enqueued = card.TryManualPlay(target);
        if (!enqueued)
        {
            return false;
        }
        // The play runs on the thread pool; pump until it finishes or blocks on a card choice.
        PumpCombatUntilIdleOrChoice();
        TryOfferCombatRewards();
        return true;
    }

    /// <summary>
    /// End the given player's turn and resolve the enemy turn to quiescence: the enemy
    /// turn runs on background tasks (not the player action queue), so we wait (via the
    /// combat's events) until the player can act again or combat ends.
    /// </summary>
    public void EndTurn(Player player)
    {
        MegaCrit.Sts2.Core.Commands.PlayerCmd.EndTurn(player, canBackOut: false);
        DrainActionQueue();
        WaitUntilPlayerCanActOrCombatEnds(player);
        TryOfferCombatRewards();
    }

    // ---------------------------------------------------------------------------------
    // Post-combat rewards. The faithful victory→rewards flow is driven by the combat UI
    // node (NCombatUi.OnCombatWon → CombatRoom.OfferRoomEndRewards), which is null in the
    // headless harness. We reproduce its logic half: once a won combat has fully ended we
    // generate the room's RewardsSet and register it with the synchronizer, then surface it
    // as GamePhase.Reward. Selecting/skipping happens through the public option API.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// If combat has just been won, generate and offer the room's end-of-combat rewards so
    /// they surface as <see cref="GamePhase.Reward"/>. Idempotent per combat room; a no-op
    /// while still in combat, on a loss, or for encounters that grant no rewards.
    /// </summary>
    private void TryOfferCombatRewards()
    {
        if (InCombat || PendingRewards is not null)
        {
            return;
        }
        if (Run.CurrentRoom is not MegaCrit.Sts2.Core.Rooms.CombatRoom room || ReferenceEquals(room, _rewardedRoom))
        {
            return;
        }
        // Only the victor gets rewards; a dead player means a loss (handled as game over).
        if (!Run.Players[0].Creature.IsAlive)
        {
            return;
        }
        if (room.Encounter is { ShouldGiveRewards: false })
        {
            return;
        }

        _rewardedRoom = room;
        Player player = Run.Players[0];

        // GenerateForRoomEnd populates the rewards and runs reward-modifying hooks but does not
        // offer them; we then register the set so the synchronizer's select/skip APIs work.
        MegaCrit.Sts2.Core.Rewards.RewardsSet set =
            Pump(MegaCrit.Sts2.Core.Commands.RewardsCmd.GenerateForRoomEnd(player, room));
        _ = RunManager.Instance.RewardsSetSynchronizer.BeginRewardsSet(set);
        PendingRewards = set;
    }

    private static T Pump<T>(System.Threading.Tasks.Task<T> task) => task.GetAwaiter().GetResult();

    // ---------------------------------------------------------------------------------
    // Public API: read state, list legal options, apply a chosen option.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Capture an immutable snapshot of the current mechanical game state. The snapshot
    /// holds no live references, so it remains valid after the game advances.
    /// </summary>
    public GameState GetState() => GameStateProjection.Capture(this);

    /// <summary>
    /// Enumerate the legal actions available to the given player in the current state.
    /// Currently covers combat (playable cards × legal targets, end turn) and map moves;
    /// other room/screen choices arrive with later milestones. Each option is valid only
    /// for the state it was listed from — apply it before advancing the game.
    /// </summary>
    public IReadOnlyList<GameOption> ListOptions(ulong playerId)
    {
        Player player = GetPlayer(playerId);
        var options = new List<GameOption>();

        // A mid-effect card choice takes precedence over everything else: until it is
        // resolved, the game is blocked and no other action can be taken.
        PendingChoice? pending = Selector.Pending;
        if (pending is not null)
        {
            return BuildChoiceOptions(player, pending);
        }

        // The Crystal Sphere event minigame: spend divinations clearing cells (then the revealed
        // items' rewards surface through the normal custom-reward screen).
        if (PendingCrystalSphere is { } minigame)
        {
            return BuildCrystalSphereOptions(player, minigame);
        }

        // The post-combat rewards screen: take rewards then proceed back to the map.
        if (PendingRewards is not null)
        {
            return BuildRewardOptions(player, PendingRewards);
        }

        // A treasure room with its chest open: take one of the offered relics or skip.
        if (HasTreasureChoice)
        {
            return BuildTreasureOptions(player);
        }

        // A rest site: choose a rest action (rest/smith/…).
        if (HasRestChoice)
        {
            return BuildRestOptions(player);
        }

        // An event room awaiting a choice (e.g. the opening Neow ancient event).
        if (HasActionableEvent)
        {
            return BuildEventOptions(player, CurrentEvent!);
        }

        if (InCombat)
        {
            MegaCrit.Sts2.Core.Entities.Players.PlayerCombatState? pcs = player.PlayerCombatState;
            if (pcs is null || pcs.Phase != MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play)
            {
                return options; // not this player's turn to act
            }

            MegaCrit.Sts2.Core.Combat.CombatState combat = Combat!;
            foreach (MegaCrit.Sts2.Core.Models.CardModel card in pcs.Hand.Cards)
            {
                if (!card.CanPlay())
                {
                    continue;
                }
                CardView view = GameStateProjection.ProjectCard(card, canPlay: true);
                if (card.TargetType == MegaCrit.Sts2.Core.Entities.Cards.TargetType.AnyEnemy)
                {
                    foreach (MegaCrit.Sts2.Core.Entities.Creatures.Creature enemy in combat.HittableEnemies)
                    {
                        if (card.IsValidTarget(enemy))
                        {
                            options.Add(GameOption.PlayCardOption(player, card, view, enemy));
                        }
                    }
                }
                else
                {
                    options.Add(GameOption.PlayCardOption(player, card, view, target: null));
                }
            }

            options.Add(GameOption.EndTurnOption(player));
            return options;
        }

        // Out of combat: map room choices.
        foreach (MegaCrit.Sts2.Core.Map.MapPoint point in GameStateProjection.ReachablePoints(Run))
        {
            options.Add(GameOption.MoveToOption(player, point.coord));
        }
        return options;
    }

    /// <summary>List options for the first (or only) player. Convenience for single-player.</summary>
    public IReadOnlyList<GameOption> ListOptions() => ListOptions(Run.Players[0].NetId);

    /// <summary>
    /// Build the options that resolve a pending card choice. Single-select choices enumerate
    /// one option per card (plus a skip when the choice allows selecting none). Multi-select
    /// choices (min &gt; 1) currently offer a single exact-minimum selection so play never
    /// blocks; full subset enumeration is future work.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildChoiceOptions(Player player, PendingChoice pending)
    {
        var options = new List<GameOption>();
        var views = pending.Options
            .Select(c => GameStateProjection.ProjectCard(c, canPlay: false))
            .ToList();

        if (pending.MinSelect <= 1)
        {
            for (int i = 0; i < pending.Options.Count; i++)
            {
                options.Add(GameOption.SelectCardsOption(
                    player, new[] { pending.Options[i] }, new[] { views[i] }));
            }
            if (pending.MinSelect == 0)
            {
                options.Add(GameOption.SelectCardsOption(
                    player,
                    System.Array.Empty<MegaCrit.Sts2.Core.Models.CardModel>(),
                    System.Array.Empty<CardView>()));
            }
        }
        else
        {
            var cards = pending.Options.Take(pending.MinSelect).ToList();
            var cardViews = views.Take(pending.MinSelect).ToList();
            options.Add(GameOption.SelectCardsOption(player, cards, cardViews));
        }
        return options;
    }

    /// <summary>
    /// Build the options for the post-combat rewards screen: one take option per not-yet-taken
    /// reward (card rewards expand to one option per offered card), plus a proceed option that
    /// leaves the screen and skips whatever is left.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildRewardOptions(Player player, MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
        var options = new List<GameOption>();
        foreach (MegaCrit.Sts2.Core.Rewards.Reward reward in set.Rewards)
        {
            if (reward.SuccessfullySelected)
            {
                continue;
            }
            switch (reward)
            {
                case MegaCrit.Sts2.Core.Rewards.GoldReward gold:
                    options.Add(GameOption.TakeRewardOption(player, gold, $"Take {gold.Amount} gold"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.PotionReward potion:
                    options.Add(GameOption.TakeRewardOption(
                        player, potion, $"Take potion {potion.Potion?.Id.Entry ?? "?"}"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.RelicReward relic:
                    options.Add(GameOption.TakeRewardOption(
                        player, relic, $"Take relic {relic.Relic?.Id.Entry ?? "?"}"));
                    break;
                case MegaCrit.Sts2.Core.Rewards.CardReward card:
                    foreach (MegaCrit.Sts2.Core.Models.CardModel offered in card.Cards)
                    {
                        CardView view = GameStateProjection.ProjectCard(offered, canPlay: false);
                        options.Add(GameOption.TakeCardRewardOption(player, card, offered, view));
                    }
                    break;
            }
        }
        options.Add(GameOption.ProceedFromRewardsOption(player));
        return options;
    }

    /// <summary>
    /// Build the options for an event room: one per unlocked, non-proceed event option, in the
    /// event's own option order (the index is preserved so <see cref="Apply"/> can resolve it
    /// against the game's <c>EventSynchronizer.ChooseLocalOption</c> seam).
    /// </summary>
    private static IReadOnlyList<GameOption> BuildEventOptions(Player player, MegaCrit.Sts2.Core.Models.EventModel ev)
    {
        var options = new List<GameOption>();
        IReadOnlyList<MegaCrit.Sts2.Core.Events.EventOption> current = ev.CurrentOptions;
        for (int i = 0; i < current.Count; i++)
        {
            MegaCrit.Sts2.Core.Events.EventOption opt = current[i];
            if (opt.IsLocked || opt.IsProceed)
            {
                continue;
            }
            options.Add(GameOption.ChooseEventOption(player, i, opt));
        }
        return options;
    }

    /// <summary>
    /// Build the options for an opened treasure room: one take option per offered relic plus a
    /// skip option. Resolved against the synchronizer's relic indices.
    /// </summary>
    private IReadOnlyList<GameOption> BuildTreasureOptions(Player player)
    {
        var options = new List<GameOption>();
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Models.RelicModel>? relics =
            TreasureSync.CurrentRelics;
        if (relics is not null)
        {
            for (int i = 0; i < relics.Count; i++)
            {
                options.Add(GameOption.TakeTreasureRelicOption(player, i, relics[i].Id.Entry));
            }
        }
        options.Add(GameOption.SkipTreasureOption(player));
        return options;
    }

    /// <summary>
    /// Build the options for a rest site: one per usable (enabled) option, carrying its live index
    /// so <see cref="Apply"/> can resolve it via <c>RestSiteSynchronizer.ChooseLocalOption</c>.
    /// Disabled options (e.g. Smith with no upgradable cards) are omitted.
    /// </summary>
    private IReadOnlyList<GameOption> BuildRestOptions(Player player)
    {
        var options = new List<GameOption>();
        System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption> rest =
            RestSiteSync.GetLocalOptions();
        for (int i = 0; i < rest.Count; i++)
        {
            if (rest[i].IsEnabled)
            {
                options.Add(GameOption.ChooseRestOption(player, i, rest[i].OptionId));
            }
        }
        return options;
    }

    /// <summary>
    /// Build the options for the Crystal Sphere minigame: one click option per still-hidden cell
    /// (cleared with the active tool), plus a switch-tool option for the tool not currently active.
    /// </summary>
    private static IReadOnlyList<GameOption> BuildCrystalSphereOptions(
        Player player, MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame)
    {
        var options = new List<GameOption>();
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell[,] cells = minigame.cells;
        for (int x = 0; x < cells.GetLength(0); x++)
        {
            for (int y = 0; y < cells.GetLength(1); y++)
            {
                if (cells[x, y].IsHidden)
                {
                    options.Add(GameOption.ClickCrystalSphereCellOption(player, x, y));
                }
            }
        }

        // Offer the tool not currently selected so the agent can switch between area/single clears.
        var big = MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Big;
        var small = MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Small;
        options.Add(GameOption.SetCrystalSphereToolOption(
            player, minigame.CrystalSphereTool == big ? small : big));
        return options;
    }

    /// <summary>
    /// Resolve a chosen option against the live game and pump to quiescence. The option
    /// must have come from <see cref="ListOptions(ulong)"/> for the current state.
    /// </summary>
    public void Apply(GameOption option)
    {
        switch (option.Kind)
        {
            case OptionKind.PlayCard:
                if (!PlayCard(option.CardModel!, option.Target))
                {
                    throw new InvalidOperationException($"Card play was rejected: {option.Description}");
                }
                break;
            case OptionKind.EndTurn:
                EndTurn(option.Player!);
                break;
            case OptionKind.MoveTo:
                MoveTo(option.MapCoord!.Value);
                break;
            case OptionKind.SelectCards:
                if (Selector.Pending is null)
                {
                    throw new InvalidOperationException("No card choice is pending to resolve.");
                }
                Selector.Resolve(option.SelectedCardModels!);
                // The effect resumes on the thread pool; pump until it finishes or blocks again.
                // A choice raised mid-event or mid-rest-action resumes that effect's task rather
                // than the combat action queue, so pump the matching surface.
                if (InCombat)
                {
                    PumpCombatUntilIdleOrChoice();
                }
                else if (_restChoiceTask is { IsCompleted: false } restTask)
                {
                    PumpRoomTaskUntilIdleOrChoice(restTask);
                    if (_restChoiceTask.IsCompleted)
                    {
                        _restChoiceTask = null;
                    }
                }
                else
                {
                    PumpEventUntilIdleOrChoice();
                }
                break;
            case OptionKind.ChooseEventOption:
                ApplyEventOption(option.Player!, option.EventOptionIndex!.Value);
                break;
            case OptionKind.TakeReward:
                TakeReward(option);
                break;
            case OptionKind.ProceedFromRewards:
                ProceedFromRewards();
                break;
            case OptionKind.TakeTreasureRelic:
                PickTreasureRelic(option.TreasureRelicIndex!.Value);
                break;
            case OptionKind.SkipTreasure:
                PickTreasureRelic(null);
                break;
            case OptionKind.ChooseRestOption:
                ChooseRestOption(option.RestOptionIndex!.Value);
                break;
            case OptionKind.ClickCrystalSphereCell:
                ClickCrystalSphereCell(option.CrystalSphereCell!.Value.Col, option.CrystalSphereCell.Value.Row);
                break;
            case OptionKind.SetCrystalSphereTool:
                SetCrystalSphereTool(option.CrystalSphereToolValue!.Value);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(option), option.Kind, "Unknown option kind.");
        }
    }

    /// <summary>
    /// Claim one reward from the pending rewards set. For a card reward, the chosen card is
    /// staged on the selector so the game's <c>GetSelectedCardReward</c> seam returns it. The
    /// reward's effects (gain gold/potion/relic, add card to deck) run on the thread pool, so
    /// we pump the selection task and then drain the action queue.
    /// </summary>
    private void TakeReward(GameOption option)
    {
        if (PendingRewards is null)
        {
            throw new InvalidOperationException("No rewards are pending to take.");
        }
        MegaCrit.Sts2.Core.Rewards.Reward reward = option.Reward
            ?? throw new InvalidOperationException("Reward option carried no reward.");

        if (option.CardModel is not null)
        {
            // Card reward: stage which of the offered cards to add to the deck.
            Selector.NextCardRewardPick = option.CardModel;
        }

        Pump(RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(reward));
        DrainActionQueue();
        // The rewards screen stays up (mirroring the in-game proceed button) even once every
        // reward is taken; the player leaves it explicitly via ProceedFromRewards.
    }

    /// <summary>
    /// Leave the rewards screen, skipping any rewards not yet taken (mirrors clicking proceed:
    /// the synchronizer marks the remaining rewards skipped and completes the set). For a
    /// post-combat set this returns to the map; for a custom (relic/event) set it resumes the
    /// effect that was suspended offering it, pumping it to quiescence.
    /// </summary>
    private void ProceedFromRewards()
    {
        if (PendingRewards is null)
        {
            throw new InvalidOperationException("Not on a rewards screen to proceed from.");
        }
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        if (!sync.IsRewardsSetCompleted(PendingRewards))
        {
            sync.SkipLocalRewardsSet();
        }
        DrainActionQueue();

        System.Threading.Tasks.TaskCompletionSource? resolve;
        lock (_rewardGate)
        {
            resolve = _customRewardResolve;
            _customRewardResolve = null;
            PendingRewards = null;
            if (resolve is not null)
            {
                _customRewardSignal = NewSignal();
            }
        }

        if (resolve is null)
        {
            return; // post-combat terminal set: the player is back on the map.
        }

        // Custom set: the offering effect is suspended in RewardsSet.Offer awaiting this; the set
        // is now completed, so unblock it and pump the effect to quiescence (it may finish, raise
        // another choice, or — in combat — continue the action queue).
        resolve.TrySetResult();
        if (_treasureExtraRewardsTask is { IsCompleted: false } treasureTask)
        {
            // The suspended effect was a treasure room's extra-rewards offer; resume it.
            PumpRoomTaskUntilIdleOrChoice(treasureTask);
            if (!CustomRewardPending)
            {
                _treasureExtraRewardsTask = null;
            }
        }
        else if (InCombat)
        {
            PumpCombatUntilIdleOrChoice();
        }
        else
        {
            PumpEventUntilIdleOrChoice();
        }
        TryOfferCombatRewards();
    }

    // ---------------------------------------------------------------------------------
    // Events. Choosing an event option runs the option's effect on a thread-pool task via
    // the game's EventSynchronizer; the harness blocks until it finishes (or surfaces a
    // mid-effect card choice). Once the event is finished — or down to only a "proceed"
    // option — the player leaves it by moving on the map (ProceedFromTerminal would drive
    // the null map UI, so we model leaving as a normal MoveTo).
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Choose an event option by its index into the live event's <c>CurrentOptions</c>, then
    /// pump until the option's effect finishes or blocks on a card choice.
    /// </summary>
    private void ApplyEventOption(Player player, int index)
    {
        RunManager.Instance.EventSynchronizer.ChooseLocalOption(index);
        PumpEventUntilIdleOrChoice();
        // The option may have entered and resolved a combat (shared events); offer its rewards.
        TryOfferCombatRewards();
    }

    /// <summary>
    /// Drive an event option's effect to completion. The effect runs as a fire-and-forget
    /// option task (see <c>EventSynchronizer.ChooseOptionForEvent</c>); we await those tasks
    /// but return early if the effect blocks on a card choice, so a blocked choice surfaces
    /// instead of deadlocking. Mirrors <see cref="PumpCombatUntilIdleOrChoice"/>.
    /// </summary>
    private void PumpEventUntilIdleOrChoice(int timeoutMs = 10000)
    {
        System.Threading.Tasks.Task optionTasks =
            RunManager.Instance.EventSynchronizer.AwaitPendingOptionTasks();
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { optionTasks, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out resolving an event option (waiting for it to finish, raise a card choice, offer rewards, or open the Crystal Sphere minigame).");
        }
        // If nothing is suspended the option finished; drain anything it enqueued.
        if (!EffectSuspended)
        {
            DrainActionQueue();
        }
    }

    /// <summary>
    /// After entering a room, if it is an event room, wait until the event has been initialized
    /// (options generated, or already finished). <c>EventModel.BeginEvent</c> runs as a
    /// fire-and-forget task, so its options may not be ready the instant room entry returns.
    /// A no-op for non-event rooms.
    /// </summary>
    private void WaitForEventReady(int timeoutMs = 10000)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            DrainActionQueue();
            MegaCrit.Sts2.Core.Models.EventModel? ev = CurrentEvent;
            if (ev is null || ev.IsFinished || ev.CurrentOptions.Count > 0)
            {
                return;
            }
            if (sw.ElapsedMilliseconds > timeoutMs)
            {
                throw new System.TimeoutException("Timed out waiting for the event room to initialize.");
            }
            System.Threading.Thread.Sleep(5);
        }
    }

    // ---------------------------------------------------------------------------------
    // Treasure rooms. Entering one calls TreasureRoomRelicSynchronizer.BeginRelicPicking
    // (populating the relics). The chest-open flow (gold + relic-added extra rewards) and the
    // relic award are normally driven by the null NTreasureRoom / NTreasureRoomRelicCollection
    // UI nodes, so the harness reproduces their logic halves. The relics then surface as
    // GamePhase.Treasure; the agent takes one or skips, after which it leaves via the map.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// If the current room is a freshly-entered treasure room, reproduce the logic half of
    /// <c>NTreasureRoom.OpenChest</c>: grant the chest's gold, then offer any relic-added extra
    /// rewards. Idempotent per room. The relic picking that follows is driven by the agent.
    /// </summary>
    private void TryOpenTreasureChest()
    {
        if (CurrentTreasureRoom is not { } room || ReferenceEquals(room, _openedTreasureRoom))
        {
            return;
        }
        _openedTreasureRoom = room;

        // Chest gold (DoNormalRewards never blocks — it just gains gold).
        _ = Pump(room.DoNormalRewards());
        DrainActionQueue();

        // Extra rewards (normally none): a relic can add reward sets, which go through
        // RewardsSet.Offer → our custom-reward gate. Run as a task and pump until it finishes or
        // suspends on that gate, so a relic-driven reward screen surfaces instead of deadlocking.
        _treasureExtraRewardsTask = room.DoExtraRewardsIfNeeded();
        PumpRoomTaskUntilIdleOrChoice(_treasureExtraRewardsTask);
        if (!CustomRewardPending)
        {
            _treasureExtraRewardsTask = null;
        }
    }

    /// <summary>
    /// Take the treasure relic at <paramref name="index"/>, or skip all relics when null. Mirrors
    /// the logic half of <c>NTreasureRoomRelicCollection</c>: the synchronizer awards relics via
    /// its <c>RelicsAwarded</c> event (consumed here to actually obtain them, since the UI node
    /// that normally does so is null). A singleplayer skip keeps the relics pending until room
    /// exit, so we end the voting explicitly to return the player to the map.
    /// </summary>
    private void PickTreasureRelic(int? index)
    {
        var sync = TreasureSync;
        System.Collections.Generic.List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult>? results = null;
        void OnAwarded(System.Collections.Generic.List<MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResult> r) => results = r;

        sync.RelicsAwarded += OnAwarded;
        try
        {
            if (index is null)
            {
                sync.SkipRelicLocally();
            }
            else
            {
                sync.PickRelicLocally(index);
            }
            // Executes the PickRelicAction → OnPicked → AwardRelics → RelicsAwarded.
            DrainActionQueue();
        }
        finally
        {
            sync.RelicsAwarded -= OnAwarded;
        }

        if (results is not null)
        {
            foreach (var result in results)
            {
                if (result.type == MegaCrit.Sts2.Core.Entities.TreasureRelicPicking.RelicPickingResultType.Skipped
                    || result.player is null)
                {
                    continue;
                }
                MegaCrit.Sts2.Core.Models.RelicModel relic = result.relic.ToMutable();
                System.Threading.Tasks.Task obtain = MegaCrit.Sts2.Core.Helpers.TaskHelper.RunSafely(
                    MegaCrit.Sts2.Core.Commands.RelicCmd.Obtain(relic, result.player));
                PumpRoomTaskUntilIdleOrChoice(obtain);
            }
        }

        // A singleplayer skip records the skip but keeps CurrentRelics until the room is exited;
        // end voting now so the player is no longer mid-pick and can move on via the map.
        if (index is null)
        {
            sync.OnRoomExited();
        }
    }

    /// <summary>
    /// Drive a fire-and-forget room/effect task until it finishes or suspends on the agent (a
    /// card choice or a custom reward screen), draining the action queue when nothing is
    /// suspended. Mirrors <see cref="PumpCombatUntilIdleOrChoice"/> for out-of-combat tasks.
    /// </summary>
    private void PumpRoomTaskUntilIdleOrChoice(System.Threading.Tasks.Task roomTask, int timeoutMs = 10000)
    {
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { roomTask, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out pumping a room task (waiting for it to finish, a card choice, rewards, or the Crystal Sphere minigame).");
        }
        if (!EffectSuspended)
        {
            DrainActionQueue();
        }
    }

    // ---------------------------------------------------------------------------------
    // Rest sites. Entering a RestSiteRoom calls RestSiteSynchronizer.BeginRestSite (generating the
    // rest/smith options). Unlike treasure, the action resolves directly through the synchronizer —
    // ChooseLocalOption runs the option's effect (heal, or Smith's deck upgrade) — so there is no
    // UI logic-half to reproduce. Smith raises a card choice through the same selector seam as
    // combat; Heal's (usually empty) reward offer goes through the custom-reward gate. After a
    // successful choice the game clears the remaining options, leaving the player on the map.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Choose the rest-site option at <paramref name="index"/> (into the synchronizer's live option
    /// list). The option's effect runs as a task; pump it until it finishes or suspends on a card
    /// choice (Smith) so the choice surfaces instead of deadlocking.
    /// </summary>
    private void ChooseRestOption(int index)
    {
        _restChoiceTask = RestSiteSync.ChooseLocalOption(index);
        PumpRoomTaskUntilIdleOrChoice(_restChoiceTask);
        if (_restChoiceTask.IsCompleted)
        {
            _restChoiceTask = null;
        }
    }

    // ---------------------------------------------------------------------------------
    // Crystal Sphere event minigame. The game drives it through a UI screen
    // (NCrystalSphereScreen) that is null headless; a Harmony patch on its ShowScreen captures the
    // plain-C# CrystalSphereMinigame here (OnCrystalSphereScreenShown) instead, suspending the
    // offering event-option task inside PlayMinigame on the minigame's completion source. The agent
    // spends each divination by clearing cells; when the last is spent the minigame completes,
    // resuming PlayMinigame, which grants the fully-revealed items' rewards via RewardsCmd.OfferCustom
    // — the same custom-reward gate as relics/events — and finishes the event.
    // ---------------------------------------------------------------------------------

    /// <summary>
    /// Spend one divination clearing the grid cell at (<paramref name="x"/>, <paramref name="y"/>)
    /// with the active tool (Big clears the surrounding 3×3 area, Small just the cell). When the last
    /// divination is spent the minigame completes; the revealed items' rewards then surface through
    /// the custom-reward screen (or, if nothing was revealed, the event simply finishes).
    /// </summary>
    private void ClickCrystalSphereCell(int x, int y)
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame =
            PendingCrystalSphere ?? throw new InvalidOperationException("No Crystal Sphere minigame is pending.");

        // CellClicked decrements the divination count, clears the cell(s), and reveals any item whose
        // footprint is now fully uncovered; on the final divination it completes the minigame's
        // completion source, resuming the suspended PlayMinigame task.
        Pump(minigame.CellClicked(minigame.cells[x, y]));

        if (!minigame.IsFinished)
        {
            DrainActionQueue();
            return;
        }

        // Last divination spent: the minigame is over. Clear our pending state, then let the resumed
        // PlayMinigame settle — it grants rewards via the custom-reward gate (which surfaces as a
        // reward screen) and then finishes the event. PumpEventUntilIdleOrChoice returns as soon as a
        // reward screen / card choice surfaces or the event-option task completes.
        lock (_crystalGate)
        {
            _pendingCrystalSphere = null;
            _crystalSphereSignal = NewSignal();
        }
        if (!EffectSuspended)
        {
            PumpEventUntilIdleOrChoice();
        }
    }

    /// <summary>Switch the Crystal Sphere divination tool (Big = 3×3 area, Small = single cell).</summary>
    private void SetCrystalSphereTool(
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType tool)
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame minigame =
            PendingCrystalSphere ?? throw new InvalidOperationException("No Crystal Sphere minigame is pending.");
        minigame.SetTool(tool);
    }

    private Player GetPlayer(ulong playerId) =>
        Run.Players.FirstOrDefault(p => p.NetId == playerId)
            ?? throw new ArgumentException($"No player with net id {playerId} in this run.", nameof(playerId));

    private enum CombatPumpResult
    {
        /// <summary>The action queue drained; nothing more to resolve.</summary>
        Idle,

        /// <summary>The current effect is blocked on a card choice (see <see cref="Selector"/>).</summary>
        ChoicePending,
    }

    /// <summary>
    /// Drive the action queue until it either drains or the running effect blocks on a card
    /// choice. The effect runs on a thread-pool continuation, so we wait on whichever happens
    /// first: the queue-drained task or the selector's pending-choice signal.
    /// </summary>
    private CombatPumpResult PumpCombatUntilIdleOrChoice(int timeoutMs = 10000)
    {
        System.Threading.Tasks.Task drain = RunManager.Instance.ActionExecutor.FinishedExecutingActions();
        int idx = System.Threading.Tasks.Task.WaitAny(
            new[] { drain, Selector.PendingSignal, CustomRewardSignal, CrystalSphereSignal }, timeoutMs);
        if (idx < 0)
        {
            throw new System.TimeoutException(
                "Timed out pumping the action queue (waiting for the queue to drain, a card choice, or rewards).");
        }
        // If the queue drained there is no suspended effect; otherwise a choice/reward is pending.
        return EffectSuspended ? CombatPumpResult.ChoicePending : CombatPumpResult.Idle;
    }

    /// <summary>Drive the game's action executor until its queue is empty.</summary>
    private static void DrainActionQueue() =>
        Pump(RunManager.Instance.ActionExecutor.FinishedExecutingActions());

    /// <summary>
    /// Block until the player is back in their Play phase or combat has ended.
    ///
    /// The enemy turn resolves on background tasks (it genuinely yields off this
    /// thread), so rather than poll we await a completion source wired to the combat's
    /// own events — it fires the instant the player can act again. The timeout is only
    /// a safety net against a hang and throws if hit (it should never be reached).
    /// </summary>
    private static void WaitUntilPlayerCanActOrCombatEnds(Player player, int timeoutMs = 5000)
    {
        var cm = MegaCrit.Sts2.Core.Combat.CombatManager.Instance;
        var tcs = new System.Threading.Tasks.TaskCompletionSource(
            System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);

        void TryComplete()
        {
            if (PlayerCanActOrCombatEnded(cm, player))
            {
                tcs.TrySetResult();
            }
        }

        void OnTurnStarted(MegaCrit.Sts2.Core.Combat.CombatState _) => TryComplete();
        void OnCombatEnded(MegaCrit.Sts2.Core.Rooms.CombatRoom _) => tcs.TrySetResult();

        // Subscribe before the first check so we cannot miss the transition.
        cm.TurnStarted += OnTurnStarted;
        cm.CombatEnded += OnCombatEnded;
        MegaCrit.Sts2.Core.Entities.Players.PlayerCombatState? pcs = player.PlayerCombatState;
        if (pcs is not null)
        {
            pcs.PlayerTurnPhaseChanged += TryComplete;
        }
        try
        {
            TryComplete(); // in case it is already satisfied
            if (!tcs.Task.Wait(timeoutMs))
            {
                throw new System.TimeoutException(
                    "Timed out waiting for the player's turn to resume or combat to end.");
            }
        }
        finally
        {
            cm.TurnStarted -= OnTurnStarted;
            cm.CombatEnded -= OnCombatEnded;
            if (pcs is not null)
            {
                pcs.PlayerTurnPhaseChanged -= TryComplete;
            }
        }

        DrainActionQueue();
    }

    private static bool PlayerCanActOrCombatEnded(MegaCrit.Sts2.Core.Combat.CombatManager cm, Player player)
    {
        if (!cm.IsInProgress)
        {
            return true;
        }
        var pcs = player.PlayerCombatState;
        return pcs is not null && pcs.Phase == MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play;
    }

    /// <summary>
    /// Drive a game <see cref="Task"/> to completion on the calling thread. The
    /// engine is async/Task-based but, in headless no-delay mode, work either runs
    /// synchronously or completes on the thread pool, so blocking here is safe.
    /// This is a placeholder for the richer quiescence pump.
    /// </summary>
    private static void Pump(System.Threading.Tasks.Task task) => task.GetAwaiter().GetResult();
}
